"""План из сведённого облака без нейросети: стены, окна, двери, комнаты.

    python -m simscan floorplan D:/scans/kvartira.e57 --out D:/plan
    -> plan.png (на уровень - plan_L1.png, plan_L2.png ...), walls.png (развёртки стен),
       elements.json

Для пустых квартир (новостройки) в формате заказчика: одно сведённое облако E57.
Структура результата - как в ReFloorBRUSNIKA (core/models.py): стены (WallSegment:
прямоугольник, толщина, наружная или нет), проёмы (Opening: door / window / gap, ширина,
стена-хозяйка) и помещения (Room: многоугольник, площадь). Координаты - метры в
выпрямленной СК плана (оси вдоль стен); как вернуться в СК файла - в elements.json.

Как это работает (на каждый уровень, найденный detect_levels):
  1. два прохода по файлу: по выборке - уровни, поворот плана и кадр; затем все точки
     раскладываются по растру: пол, потолок, полосы высоты и битовая маска слоёв по 10 см
     (бит k - в пикселе есть точки на высоте k*10 см над полом);
  2. грани стен - вертикальные поверхности: точки минимум в двух из трёх полос 1,0-1,8 /
     2,15-2,5 / 2,5 м - потолок. Горизонтальное (низкий потолок за дверью, светильник)
     отсеивается, перемычки над дверями остаются и замыкают помещения. Толща стены -
     пиксели между двумя гранями не дальше 45 см;
  3. граф стен по скелету: узлы - углы и примыкания (общие - стены стыкуются точно),
     рёбра - осевые ломаные (дуговая стена - несколько кусков). Хвосты и острова мусора
     отбрасываются, свободные концы пристыковываются к ближайшей стене;
  4. проёмы - по развёртке каждой стены (длина x слои по 10 см): участок без точек на
     1,0-1,8 м. Подоконник - верх точек ниже, перемычка - низ точек выше: подоконник есть -
     окно, нет и есть перемычка - дверь, нет ничего - проём во всю высоту;
  5. помещения - здание (пол или потолок виден), разрезанное стенами.
Наружная толщина не видна (грань одна) - на плане она условная.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

BANDS = {"low": (0.15, 0.7), "mid": (1.0, 1.8)}
LAYER_M = 0.1


@dataclass
class WallSegment:
    id: str
    points: list                      # осевая линия - ломаная, метры (дуговая стена - несколько кусков)
    start_node: str                   # узлы общие для стыкующихся стен
    end_node: str
    thickness: float | None          # None - наружная, толщина не видна
    is_outer: bool
    length: float


@dataclass
class Opening:
    id: str
    opening_type: str                 # door | window | gap
    wall_id: str
    a: list                           # начало и конец по оси стены, метры
    b: list
    points: list                      # четырёхугольник проёма
    width: float
    sill: float | None = None         # низ проёма над полом (окно)
    head: float | None = None         # верх проёма над полом


@dataclass
class Room:
    id: str
    name: str
    points: list                      # многоугольник, метры
    area: float
    perimeter: float
    label_xy: tuple = (0.0, 0.0)


@dataclass
class LevelPlan:
    level: int
    floor_z: float
    ceiling_z: float
    walls: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    rooms: list = field(default_factory=list)
    nodes: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Чтение: два прохода
# ----------------------------------------------------------------------

def _quat_R(q) -> np.ndarray:
    from .inspect_e57 import _quat_R as qr

    return qr(q)


def _scan_points(reader, index: int, chunk: int):
    s = reader.scans[index]
    R = _quat_R(s.rotation) if s.rotation is not None else np.eye(3)
    t = np.asarray(s.translation, float) if s.translation is not None else np.zeros(3)
    names = s.field_names
    cart = all(n in names for n in ("cartesianX", "cartesianY", "cartesianZ"))
    fields = ["cartesianX", "cartesianY", "cartesianZ"] if cart else \
        ["sphericalRange", "sphericalAzimuth", "sphericalElevation"]
    inv_name = "cartesianInvalidState" if cart else "sphericalInvalidState"
    if inv_name in names:
        fields.append(inv_name)
    for d in reader.iter_points(index, fields, chunk):
        if cart:
            xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
        else:
            r, az, el = d["sphericalRange"], d["sphericalAzimuth"], d["sphericalElevation"]
            xyz = np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]
        ok = np.isfinite(xyz).all(1)
        if inv_name in d:
            ok &= d[inv_name] == 0
        yield xyz[ok] @ R.T + t


def dominant_angle(xy: np.ndarray, bin_m: float = 0.02, max_points: int = 400_000) -> float:
    """Главное направление стен (градусы, 0..90): угол, при котором проекции точек среза на
    оси x и y дают самые острые пики (стены вдоль осей собираются в узкие столбцы).
    Грубо через 1°, затем уточнение через 0,05°."""
    if len(xy) > max_points:
        xy = xy[np.random.default_rng(0).choice(len(xy), max_points, replace=False)]
    xy = xy - xy.mean(0)

    def score(deg):
        t = math.radians(deg)
        c, s_ = math.cos(t), math.sin(t)
        tot = 0.0
        for v in (xy[:, 0] * c + xy[:, 1] * s_, -xy[:, 0] * s_ + xy[:, 1] * c):
            h = np.bincount(np.floor((v - v.min()) / bin_m).astype(np.int64)).astype(float)
            tot += float((h * h).sum())
        return tot

    coarse = np.arange(0.0, 90.0, 1.0)
    best = coarse[int(np.argmax([score(d) for d in coarse]))]
    fine = np.arange(best - 1.0, best + 1.0, 0.05)
    return float(fine[int(np.argmax([score(d) for d in fine]))] % 90.0)


@dataclass
class Rasters:
    origin: np.ndarray                # (x0, y0) в выпрямленной СК
    px: float
    shape: tuple
    counts: dict                      # полоса -> int32 (H, W)
    bits: np.ndarray                  # int32 (H, W): бит k - есть точки в слое высоты k
    layer_m: float = 0.1              # толщина слоя
    layers: int = 30


def build_rasters(path, px: float = 0.01, chunk: int = 2_000_000, sample_points: int = 3_000_000,
                  log=print) -> tuple[list, float, list]:
    from .e57read import E57Reader
    from .rasterize import detect_levels, floor_ceiling

    with E57Reader(path) as r:
        total = sum(s.points for s in r.scans)
        p_keep = min(1.0, sample_points / max(total, 1))
        rng = np.random.default_rng(0)
        sample = []
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                sample.append(p[rng.random(len(p)) < p_keep].astype(np.float32))
        sample = np.concatenate(sample).astype(float)
        levels = detect_levels(sample)
        if not levels:
            fl, ce = floor_ceiling(sample[:, 2])
            levels = [{"floor_z": fl, "ceiling_z": ce, "height_m": ce - fl}]
        hz = sample[:, 2] - levels[0]["floor_z"]
        angle = dominant_angle(sample[(hz > 1.0) & (hz < 1.8), :2])
        c, s_ = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
        rot = np.array([[c, -s_], [s_, c]])
        q = sample[:, :2] @ rot.T
        lo = np.percentile(q, 0.2, axis=0) - 0.6
        hi = np.percentile(q, 99.8, axis=0) + 0.6
        px = max(px, float((hi - lo).max()) / 4000)
        shape = (int(math.ceil((hi[1] - lo[1]) / px)), int(math.ceil((hi[0] - lo[0]) / px)))
        log(f"уровней {len(levels)}, поворот {angle:.2f}°, растр {shape[1]}x{shape[0]} px по {px * 1000:.0f} мм")
        rasters = []
        for lv in levels:
            rasters.append(Rasters(lo, px, shape,
                                   {k: np.zeros(shape, np.int32) for k in ("floor", "ceiling", "low", "mid", "high", "wb", "wc")},
                                   np.zeros(shape, np.int32), LAYER_M,
                                   int(min(31, math.ceil((lv["ceiling_z"] - lv["floor_z"]) / LAYER_M)))))
        H, W = shape
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                q = p[:, :2] @ rot.T
                jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                inside = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
                for lv, ras in zip(levels, rasters):
                    h = p[:, 2] - lv["floor_z"]
                    top = lv["ceiling_z"] - lv["floor_z"]
                    m_lv = inside & (h > -0.1) & (h < top + 0.1)
                    flat = ii * W + jj
                    # «потолок» - любая точка выше 2 м: подвесные потолки и короба ниже основного
                    bands = {"floor": np.abs(h) < 0.04, "ceiling": (h > 2.0) & (h < top + 0.06),
                             "low": (h > BANDS["low"][0]) & (h < BANDS["low"][1]),
                             "mid": (h > BANDS["mid"][0]) & (h < BANDS["mid"][1]),
                             "high": (h > 1.85) & (h < top - 0.06),
                             "wb": (h > 2.15) & (h < 2.5), "wc": (h > 2.5) & (h < top - 0.1)}
                    for k, m in bands.items():
                        m = m & m_lv
                        ras.counts[k] += np.bincount(flat[m], minlength=H * W).reshape(H, W).astype(np.int32)
                    # слои по 10 см: битовая маска на пиксель - развёртка стены по высоте
                    lay = np.floor(h / ras.layer_m).astype(np.int64)
                    bits = ras.bits.reshape(-1)
                    for kk in range(ras.layers):
                        m = m_lv & (lay == kk)
                        if m.any():
                            bits[np.unique(flat[m])] |= np.int32(1 << kk)
    return levels, angle, rasters


# ----------------------------------------------------------------------
# Разбор уровня
# ----------------------------------------------------------------------

def _disk(r_px: float):
    import cv2

    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    inv = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = mask.shape
    out = mask.copy()
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):
            out[lab == k] = 1
    return out


def _rect_kernel(w: int, h: int):
    return np.ones((max(1, h), max(1, w)), np.uint8)


def _overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _components(mask, conn=8):
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), conn)
    return n, lab, stats


def _sandwiched(faces: np.ndarray, L: int) -> np.ndarray:
    """Пиксели между двумя гранями не дальше L по x или по y (толща стены)."""
    import cv2

    out = np.zeros_like(faces)
    for kw, kh, a1, a2 in ((L, 1, (0, 0), (L - 1, 0)), (1, L, (0, 0), (0, L - 1))):
        k_ = _rect_kernel(kw, kh)
        out |= cv2.dilate(faces, k_, anchor=a1) & cv2.dilate(faces, k_, anchor=a2)
    return out


def _drop_small(mask: np.ndarray, min_extent_px: int, min_area_px: int = 0) -> np.ndarray:
    n, lab, stats = _components(mask)
    bad = np.flatnonzero((np.maximum(stats[:, 2], stats[:, 3]) < min_extent_px) | (stats[:, 4] < min_area_px))
    out = mask.copy()
    out[np.isin(lab, bad[bad > 0])] = 0
    return out


def _thin(mask: np.ndarray) -> np.ndarray:
    """Скелет (Чжан - Суэнь) в рамке маски."""
    ys, xs = np.nonzero(mask)
    out = np.zeros(mask.shape, np.uint8)
    if not len(ys):
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P_ = np.pad(img, 1)
            p2, p3, p4, p5 = P_[:-2, 1:-1], P_[:-2, 2:], P_[1:-1, 2:], P_[2:, 2:]
            p6, p7, p8, p9 = P_[2:, 1:-1], P_[2:, :-2], P_[1:-1, :-2], P_[:-2, :-2]
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                c = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            break
    out[y0:y1, x0:x1] = img
    return out


_NB = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


class WallGraph:
    """Граф стен: узлы - углы, примыкания и концы (общие для всех рёбер - стены стыкуются
    точно), рёбра - осевые линии стен (ломаные: дуговые стены идут несколькими кусками)."""

    def __init__(self):
        self.nodes: dict[int, np.ndarray] = {}
        self.edges: dict[int, dict] = {}
        self._nid = 0
        self._eid = 0

    def add_node(self, xy) -> int:
        self._nid += 1
        self.nodes[self._nid] = np.asarray(xy, float)
        return self._nid

    def add_edge(self, a, b, pts, **kw) -> int:
        self._eid += 1
        pts = np.asarray(pts, float).copy()
        pts[0], pts[-1] = self.nodes[a], self.nodes[b]
        self.edges[self._eid] = {"a": a, "b": b, "pts": pts, **kw}
        return self._eid

    def degree(self, n) -> int:
        return sum((e["a"] == n) + (e["b"] == n) for e in self.edges.values())

    def incident(self, n) -> list:
        return [k for k, e in self.edges.items() if n in (e["a"], e["b"])]

    @staticmethod
    def length(e) -> float:
        return float(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1).sum())

    def drop_unused_nodes(self):
        used = {e["a"] for e in self.edges.values()} | {e["b"] for e in self.edges.values()}
        self.nodes = {k: v for k, v in self.nodes.items() if k in used}

    def oriented(self, k, start):
        e = self.edges[k]
        return e["pts"] if e["a"] == start else e["pts"][::-1]

    def merge_degree2(self):
        """Узел с двумя рёбрами - не узел: рёбра склеиваются в одно."""
        changed = True
        while changed:
            changed = False
            for n in list(self.nodes):
                inc = self.incident(n)
                if len(inc) != 2 or inc[0] == inc[1]:
                    continue
                e1, e2 = self.edges[inc[0]], self.edges[inc[1]]
                a = e1["b"] if e1["a"] == n else e1["a"]
                b = e2["b"] if e2["a"] == n else e2["a"]
                p1 = self.oriented(inc[0], a)
                p2 = self.oriented(inc[1], n)
                w1, w2 = self.length(e1), self.length(e2)
                kw = {"t": (e1["t"] * w1 + e2["t"] * w2) / max(w1 + w2, 1e-9)}
                del self.edges[inc[0]], self.edges[inc[1]]
                self.add_edge(a, b, np.vstack([p1, p2[1:]]), **kw)
                del self.nodes[n]
                changed = True
                break

    def split(self, k, xy) -> int:
        """Разбить ребро k в ближайшей к xy точке; вернуть новый узел."""
        e = self.edges[k]
        pts = e["pts"]
        best = (1e18, 0, pts[0])
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            d = b - a
            t = np.clip(np.dot(np.asarray(xy) - a, d) / max(np.dot(d, d), 1e-12), 0, 1)
            q = a + t * d
            dist = np.linalg.norm(q - xy)
            if dist < best[0]:
                best = (dist, i, q)
        _, i, q = best
        for end, node in ((pts[0], e["a"]), (pts[-1], e["b"])):
            if np.linalg.norm(q - end) < 1.0:
                return node
        n = self.add_node(q)
        del self.edges[k]
        self.add_edge(e["a"], n, np.vstack([pts[:i + 1], q]), t=e["t"])
        self.add_edge(n, e["b"], np.vstack([q, pts[i + 1:]]), t=e["t"])
        return n


def _trace_graph(skel: np.ndarray, dist: np.ndarray) -> WallGraph:
    import cv2

    H, W = skel.shape
    nb = cv2.filter2D(skel, cv2.CV_16S, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - skel
    node_mask = (skel > 0) & (nb != 2)
    n_cl, cl = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    g = WallGraph()
    node_of = {}
    for c in range(1, n_cl):
        ys, xs = np.nonzero(cl == c)
        node_of[c] = g.add_node((xs.mean(), ys.mean()))
    visited = np.zeros_like(skel, bool)

    def nbrs(y, x):
        for dy, dx in _NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
                yield yy, xx

    for y, x in zip(*np.nonzero(node_mask)):
        start = node_of[cl[y, x]]
        for ny, nx in nbrs(y, x):
            if node_mask[ny, nx] or visited[ny, nx]:
                continue
            path = [(x, y), (nx, ny)]
            visited[ny, nx] = True
            prev, cur, end = (y, x), (ny, nx), None
            while end is None:
                nxt = None
                for q in nbrs(*cur):
                    if q == prev:
                        continue
                    if node_mask[q]:
                        if cl[q] != cl[y, x] or len(path) > 3:
                            nxt, end = q, node_of[cl[q]]
                            break
                        continue
                    if not visited[q]:
                        nxt = q
                        break
                if nxt is None:
                    break
                path.append((nxt[1], nxt[0]))
                if end is None:
                    visited[nxt] = True
                prev, cur = cur, nxt
            if end is None:
                continue                             # обрыв без узла - не бывает, но не падаем
            p = np.array(path, float)
            t = 2 * float(np.median(dist[p[:, 1].astype(int), p[:, 0].astype(int)])) - 1
            g.add_edge(start, end, p, t=max(t, 1.0))
    return g


def _simplify(pts: np.ndarray, eps: float) -> np.ndarray:
    import cv2

    if len(pts) <= 2:
        return pts
    s = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, False)[:, 0, :].astype(float)
    s[0], s[-1] = pts[0], pts[-1]
    return s


def _seg_poly(a, b, t, ext):
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        return None
    d = d / L
    n = np.array([-d[1], d[0]])
    a2, b2 = a - d * ext, b + d * ext
    return np.array([a2 + n * t / 2, b2 + n * t / 2, b2 - n * t / 2, a2 - n * t / 2])


def _polyline_poly(pts, t, ext):
    """Многоугольник стены по осевой ломаной: стыки внутри - со срезом под углом (без
    зубцов), концы продлены на ext (перекрывают узел)."""
    pts = np.asarray(pts, float).copy()
    d = np.diff(pts, axis=0)
    ln = np.linalg.norm(d, axis=1)
    keep = np.r_[True, ln > 1e-9]
    pts, d = pts[keep], d[ln > 1e-9]
    if len(pts) < 2:
        return None
    d = d / np.linalg.norm(d, axis=1)[:, None]
    pts[0] -= d[0] * ext
    pts[-1] += d[-1] * ext
    nrm = np.c_[-d[:, 1], d[:, 0]]
    left, right = [], []
    for i, p in enumerate(pts):
        if i == 0:
            v = nrm[0]
        elif i == len(pts) - 1:
            v = nrm[-1]
        else:
            m = nrm[i - 1] + nrm[i]
            m = m / max(np.linalg.norm(m), 1e-9)
            v = m / max(float(np.dot(m, nrm[i])), 0.5)
        left.append(p + v * t / 2)
        right.append(p - v * t / 2)
    return np.array(left + right[::-1])


def _polyline_at(pts, s):
    """Точка и направление ломаной на длине s от начала."""
    seg = np.diff(pts, axis=0)
    ls = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(ls)]
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    d = seg[i] / max(ls[i], 1e-9)
    return pts[i] + d * (s - cum[i]), d


def analyse_level(ras: Rasters, lv: dict, k: int, max_wall_m: float = 0.45, outer_draw_m: float = 0.2,
                  max_opening_m: float = 2.0, min_opening_m: float = 0.5, snap_m: float = 0.35,
                  spur_m: float = 0.3, min_island_m: float = 1.0, min_room_m2: float = 1.0
                  ) -> tuple[LevelPlan, dict]:
    import cv2

    px = ras.px
    H, W = ras.shape
    P = lambda m: max(1, int(round(m / px)))                      # noqa: E731 - метры -> пиксели
    occ = {b: (ras.counts[b] > 0).astype(np.uint8) for b in ras.counts}
    k3 = _rect_kernel(3, 3)

    # --- здание: где виден пол или потолок, дыры закрыты ---------------------------------
    free = cv2.morphologyEx(occ["floor"] | occ["ceiling"], cv2.MORPH_CLOSE, k3)
    building = _fill_holes(cv2.morphologyEx(free, cv2.MORPH_CLOSE, _rect_kernel(P(0.3), P(0.3))))
    building = cv2.morphologyEx(building, cv2.MORPH_OPEN, _rect_kernel(P(0.3), P(0.3)))

    # --- грани стен: вертикальная поверхность - точки минимум в двух из трёх полос
    # (1,0-1,8 / 2,15-2,5 / 2,5 м - потолок). Горизонтальное (низкий потолок за дверью,
    # светильник, провод) попадает в одну полосу и отсеивается; перемычка над дверью
    # (две верхние полосы) остаётся - она замыкает помещения ---------------------------------
    votes = sum(cv2.dilate(occ[b], k3) for b in ("mid", "wb", "wc"))
    faces = ((votes >= 2) & (occ["mid"] | occ["wb"] | occ["wc"])).astype(np.uint8)
    faces &= cv2.dilate(building, _rect_kernel(2 * P(0.15) + 1, 2 * P(0.15) + 1))
    faces = _drop_small(cv2.morphologyEx(faces, cv2.MORPH_CLOSE, k3), P(0.25))
    struct = faces | _sandwiched(faces, P(max_wall_m))          # толща: между двумя гранями
    struct = _drop_small(cv2.morphologyEx(struct, cv2.MORPH_CLOSE, k3), P(0.3))
    dist = cv2.distanceTransform(struct, cv2.DIST_L2, 5)

    # --- граф стен по скелету -------------------------------------------------------------
    g = _trace_graph(_thin(struct), dist)
    # хвосты (наплывы, откосы, мусор у стены) - обрезать, пока есть
    changed = True
    while changed:
        changed = False
        for kk, e in list(g.edges.items()):
            da, db = g.degree(e["a"]), g.degree(e["b"])
            if (min(da, db) == 1 and max(da, db) != 1
                    and g.length(e) < max(P(spur_m), 1.5 * e["t"])):
                del g.edges[kk]
                changed = True
        g.drop_unused_nodes()
        g.merge_degree2()
    # острова: связные куски короче min_island_m - мусор (светильник, откос, соседний дом)
    comp = {n: n for n in g.nodes}

    def find(n):
        while comp[n] != n:
            comp[n] = comp[comp[n]]
            n = comp[n]
        return n

    for e in g.edges.values():
        comp[find(e["a"])] = find(e["b"])
    total = {}
    for e in g.edges.values():
        total[find(e["a"])] = total.get(find(e["a"]), 0) + g.length(e)
    for kk, e in list(g.edges.items()):
        if total[find(e["a"])] < P(min_island_m):
            del g.edges[kk]
    g.drop_unused_nodes()

    # --- несостыковки: свободный конец до соседней стены не дальше snap_m - пристыковать;
    # дальше (до max_opening_m) по направлению стены - тоже, это проём во всю высоту ------
    def edge_raster(skip=None):
        r = np.zeros((H, W), np.int32)
        for kk, e in g.edges.items():
            if kk != skip:
                cv2.polylines(r, [np.round(e["pts"]).astype(np.int32)], False, int(kk),
                              max(1, int(round(e["t"]))))
        return r

    for n in list(g.nodes):
        if n not in g.nodes or g.degree(n) != 1:
            continue
        (kk,) = g.incident(n)
        e = g.edges[kk]
        pts = g.oriented(kk, n)                          # от конца внутрь стены
        far = min(len(pts) - 1, 1)
        d = pts[0] - pts[far]
        d = d / max(np.linalg.norm(d), 1e-9)
        r = edge_raster(skip=kk)
        hit = None
        p0 = pts[0]
        # по направлению стены
        for s in range(1, P(max_opening_m) + 1):
            q = np.round(p0 + d * s).astype(int)
            if not (0 <= q[0] < W and 0 <= q[1] < H):
                break
            if r[q[1], q[0]]:
                hit = (int(r[q[1], q[0]]), p0 + d * s)
                break
        # или вбок к ближайшей
        if hit is None:
            ys, xs = np.nonzero(r)
            if len(xs):
                dd = np.hypot(xs - p0[0], ys - p0[1])
                i = int(dd.argmin())
                if dd[i] <= P(snap_m) + e["t"] / 2:
                    hit = (int(r[ys[i], xs[i]]), np.array([xs[i], ys[i]], float))
        if hit is None or hit[0] not in g.edges:
            continue
        m = g.split(hit[0], hit[1])
        g.add_edge(n, m, np.vstack([p0, g.nodes[m]]), t=e["t"])
    g.merge_degree2()
    # --- наружные: с одной стороны вне здания ----------------------------------------------
    outside = (1 - building).astype(np.uint8)
    for e in g.edges.values():
        L = g.length(e)
        sides = [[], []]
        for s in np.linspace(0, L, max(3, int(L / P(0.2)))):
            q, d = _polyline_at(e["pts"], s)
            nrm = np.array([-d[1], d[0]])
            for si, sg in enumerate((1, -1)):
                z = np.round(q + sg * nrm * (e["t"] / 2 + P(0.15))).astype(int)
                sides[si].append(1 if not (0 <= z[0] < W and 0 <= z[1] < H) else outside[z[1], z[0]])
        e["outer"] = max(np.mean(sides[0]), np.mean(sides[1])) > 0.6
        # ось - ломаная: дуга остаётся дугой (куски по 3 см отклонения); у наружных грубее -
        # откосы окон не дают зубцов
        e["pts"] = _simplify(e["pts"], P(0.1) if e["outer"] else P(0.03))

    # --- проёмы по развёртке стены: вдоль оси - позиция, вверх - слои по 10 см; в каждой
    # позиции слой занят, если в сечении стены (толщина + 6 см) есть точки этого слоя ---------
    openings = []
    elevations = {}
    nl, lm = ras.layers, ras.layer_m
    z_of = lambda kk: kk * lm                                    # noqa: E731
    mid_l = slice(int(1.0 / lm), int(1.8 / lm))
    for kk, e in g.edges.items():
        L = g.length(e)
        if L < P(0.5):
            continue
        half = e["t"] / 2 + P(0.06)
        ss = np.arange(0, L, 1.0)
        acc = np.zeros(len(ss), np.int64)
        offs = np.arange(-half, half + 1, 1.0)
        for i, s_ in enumerate(ss):
            q, d = _polyline_at(e["pts"], s_)
            z = np.round(q[None] + np.array([-d[1], d[0]])[None] * offs[:, None]).astype(int)
            ok = (z[:, 0] >= 0) & (z[:, 0] < W) & (z[:, 1] >= 0) & (z[:, 1] < H)
            acc[i] = np.bitwise_or.reduce(ras.bits[z[ok, 1], z[ok, 0]]) if ok.any() else 0
        E = ((acc[:, None] >> np.arange(nl)[None]) & 1).astype(bool)          # (позиция, слой)
        elevations[kk] = E
        if L < P(min_opening_m):
            continue
        kw = max(1, P(0.04))
        mid_occ = np.convolve(E[:, mid_l].mean(1), np.ones(kw) / kw, "same") > 0.3
        runs = np.flatnonzero(np.diff(np.r_[0, (~mid_occ).astype(int), 0]))
        for a, b in zip(runs[::2], runs[1::2]):
            if not (P(min_opening_m) <= b - a <= P(max_opening_m + 0.5)):
                continue
            v = E[a:b].mean(0) > 0.3                     # профиль проёма по высоте
            below = np.flatnonzero(v[1:mid_l.start]) + 1     # слой 0 - пол, не в счёт
            above = np.flatnonzero(v[mid_l.stop:nl - 1]) + mid_l.stop
            sill = z_of(below.max() + 1) if len(below) else 0.0
            head = z_of(above.min()) if len(above) else None
            if sill >= 0.25:
                kind = "window"
            elif head is not None:
                kind = "door"
            else:
                kind = "window" if e["outer"] else "gap"
            p_a, _ = _polyline_at(e["pts"], ss[a])
            p_b, _ = _polyline_at(e["pts"], ss[b - 1])
            quad = _seg_poly(p_a, p_b, max(e["t"], P(outer_draw_m) if e["outer"] else 1.0), 0)
            openings.append({"type": kind, "edge": kk, "quad": quad, "a": p_a, "b": p_b,
                             "s": (ss[a] * px, ss[b - 1] * px), "width": (ss[b - 1] - ss[a] + 1) * px,
                             "sill": sill if kind == "window" else None, "head": head})

    # один проём на двух соседних рёбрах (облицовка, двойная грань) - оставить на толстом
    openings.sort(key=lambda o: -g.edges[o["edge"]]["t"])
    kept = []
    for o in openings:
        c = (o["a"] + o["b"]) / 2
        if all(np.linalg.norm(c - (q["a"] + q["b"]) / 2) > max(o["width"], q["width"]) / px / 2 for q in kept):
            kept.append(o)
    openings = kept

    # --- помещения: здание без стен; перемычки и проёмы замыкают контуры -----------------
    cut = struct.copy()
    for e in g.edges.values():
        t = max(e["t"], 2.0)
        poly = _polyline_poly(e["pts"], t, t / 2)
        if poly is not None:
            cv2.fillPoly(cut, [np.round(poly).astype(np.int32)], 1)
    rooms_mask = (building & (1 - cut)).astype(np.uint8)
    rooms_mask = cv2.morphologyEx(rooms_mask, cv2.MORPH_OPEN, _rect_kernel(P(0.1), P(0.1)))
    n, lab, stats = _components(rooms_mask, 4)
    rooms = []
    for c in range(1, n):
        area = stats[c][4] * px * px
        if area < min_room_m2:
            continue
        m = (lab == c).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, P(0.03), True)[:, 0, :]
        dm = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        li, lj = np.unravel_index(dm.argmax(), dm.shape)
        rooms.append({"poly_px": poly, "area": area, "perimeter": cv2.arcLength(poly.reshape(-1, 1, 2), True) * px,
                      "label_px": (lj, li)})

    # --- в метры (выпрямленная СК, y вверх) -----------------------------------------------
    def to_m(xy):
        return [round(float(ras.origin[0] + xy[0] * px), 3), round(float(ras.origin[1] + (H - xy[1]) * px), 3)]

    plan = LevelPlan(k, lv["floor_z"], lv["ceiling_z"])
    node_id = {n: f"L{k}-N{i + 1}" for i, n in enumerate(sorted(g.nodes))}
    plan.nodes = [{"id": node_id[n], "xy": to_m(v), "degree": g.degree(n)} for n, v in sorted(g.nodes.items())]
    wall_id = {}
    for i, (kk, e) in enumerate(sorted(g.edges.items())):
        wall_id[kk] = f"L{k}-W{i + 1}"
        plan.walls.append(WallSegment(wall_id[kk], [to_m(p) for p in e["pts"]], node_id[e["a"]], node_id[e["b"]],
                                      None if e["outer"] else round(e["t"] * px, 3), bool(e["outer"]),
                                      round(g.length(e) * px, 3)))
    for i, o in enumerate(openings):
        plan.openings.append(Opening(f"L{k}-O{i + 1}", o["type"], wall_id[o["edge"]], to_m(o["a"]), to_m(o["b"]),
                                     [to_m(p) for p in o["quad"]], round(o["width"], 3),
                                     None if o["sill"] is None else round(o["sill"], 3),
                                     None if o["head"] is None else round(o["head"], 3)))
    for i, rm in enumerate(sorted(rooms, key=lambda r: -r["area"])):
        plan.rooms.append(Room(f"L{k}-R{i + 1}", f"Помещение {i + 1}", [to_m(p) for p in rm["poly_px"]],
                               round(rm["area"], 2), round(rm["perimeter"], 2), tuple(to_m(rm["label_px"]))))
    masks = {"free": free, "building": building, "faces": faces, "struct": struct,
             "elevations": [(wall_id[kk], elevations[kk], [o for o in openings if o["edge"] == kk])
                            for kk in sorted(elevations)]}
    return plan, masks


# ----------------------------------------------------------------------
# Рисунок
# ----------------------------------------------------------------------

ROOM_TINTS = ("#dce9f8", "#fbe3d6", "#d6f0e5", "#fbefcc", "#f8e1ea", "#d9ecd9", "#e3dff3", "#f8dcdc")
WALL, WALL_OUT = "#3a3a3a", "#141414"


def render(plan: LevelPlan, ras: Rasters, masks: dict, path: Path, title: str = "") -> None:
    from matplotlib.patches import Arc, Patch, Polygon

    from .debug import AQUA, INK, INK2, ORANGE, SURFACE, _plt

    plt = _plt()
    xy = np.array([p for w in plan.walls for p in w.points] + [p for r in plan.rooms for p in r.points])
    if not len(xy):
        xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    xlim = (xy[:, 0].min() - 0.4, xy[:, 0].max() + 0.4)
    ylim = (xy[:, 1].min() - 0.4, xy[:, 1].max() + 0.4)
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig = plt.figure(figsize=(11 * span_x / max(span_x, span_y) + 4.2, 11 * span_y / max(span_x, span_y) + 0.6))
    ax = fig.add_axes([0.02, 0.03, 0.70, 0.94])
    side = fig.add_axes([0.74, 0.03, 0.25, 0.94])
    side.axis("off")
    ax.set_facecolor(SURFACE)
    for i, rm in enumerate(plan.rooms):
        ax.add_patch(Polygon(rm.points, closed=True, facecolor=ROOM_TINTS[i % len(ROOM_TINTS)],
                             edgecolor="none", zorder=1))
        ax.text(*rm.label_xy, f"{i + 1}\n{rm.area:.1f} м²".replace(".", ","), ha="center", va="center",
                fontsize=10, color=INK, zorder=6)
    for w in plan.walls:
        t = w.thickness if w.thickness else 0.2
        color = WALL_OUT if w.is_outer else WALL
        poly = _polyline_poly(w.points, t, t / 2)
        if poly is not None:
            ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, lw=0.2, zorder=2))
    for o in plan.openings:
        a, b = np.array(o.a), np.array(o.b)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        nrm = np.array([-d[1], d[0]])
        ax.add_patch(Polygon(o.points, closed=True, facecolor=SURFACE, edgecolor="none", zorder=3))
        if o.opening_type == "window":
            q = np.array(o.points)
            t = np.linalg.norm(q[0] - q[3])
            for f in (-0.2, 0.2):
                ax.plot(*np.c_[a + nrm * t * f, b + nrm * t * f], color=AQUA, lw=1.4, zorder=4)
            ax.add_patch(Polygon(o.points, closed=True, fill=False, edgecolor=AQUA, lw=0.8, zorder=4))
        elif o.opening_type == "door":
            r = o.width
            ang = math.degrees(math.atan2(d[1], d[0]))
            ax.plot(*np.c_[a, a + nrm * r], color=ORANGE, lw=1.4, zorder=4)
            ax.add_patch(Arc(tuple(a), 2 * r, 2 * r, theta1=ang, theta2=ang + 90, color=ORANGE, lw=1.0,
                             zorder=4))
        c = (a + b) / 2 - nrm * 0.22
        rot = math.degrees(math.atan2(d[1], d[0]))
        rot = rot - 180 if rot > 90 else rot + 180 if rot < -90 else rot
        ax.text(*c, f"{o.width * 1000:.0f}", fontsize=7, color=INK2, ha="center", va="center", rotation=rot,
                zorder=6)
    sx, sy = xlim[0] + 0.3, ylim[0] + 0.15
    ax.plot([sx, sx + 1], [sy, sy], color=INK, lw=2)
    ax.text(sx + 0.5, sy + 0.08, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    nd = sum(o.opening_type == "door" for o in plan.openings)
    nw = sum(o.opening_type == "window" for o in plan.openings)
    ng = sum(o.opening_type == "gap" for o in plan.openings)
    lines = [title, f"пол {plan.floor_z:+.2f} м, потолок {plan.ceiling_z - plan.floor_z:.2f} м над полом",
             f"стен {len(plan.walls)}, узлов {len(plan.nodes)}", f"дверей {nd}, окон {nw}, проёмов {ng}", ""]
    for i, rm in enumerate(plan.rooms):
        lines.append(f"{i + 1:>2}  {rm.area:6.1f} м²   периметр {rm.perimeter:5.1f} м".replace(".", ","))
    lines.append(f"    {sum(r.area for r in plan.rooms):6.1f} м²   всего".replace(".", ","))
    lines.append("")
    for o in plan.openings:
        kind = {"door": "дверь", "window": "окно", "gap": "проём"}[o.opening_type]
        hs = []
        if o.sill is not None:
            hs.append(f"низ {o.sill:.2f}")
        if o.head is not None:
            hs.append(f"верх {o.head:.2f}")
        lines.append(f"{kind:<6} {o.width * 1000:5.0f} мм  {' '.join(hs)}".replace(".", ","))
    side.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace", color=INK)
    side.legend(handles=[Patch(color=WALL, label="стена"),
                         Patch(color=WALL_OUT, label="наружная (толщина условная)"),
                         Patch(facecolor=SURFACE, edgecolor=AQUA, label="окно"),
                         Patch(color=ORANGE, label="дверь")],
                loc="lower left", frameon=False, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_elevations(items, px: float, layer_m: float, path: Path, min_len_m: float = 1.0) -> None:
    """Развёртки стен: по горизонтали - длина вдоль стены, по вертикали - слои по 10 см;
    тёмное - точки есть. Рамки - найденные проёмы."""
    from matplotlib.patches import Rectangle

    from .debug import AQUA, INK, INK2, ORANGE, _plt

    plt = _plt()
    items = [it for it in items if len(it[1]) * px >= min_len_m]
    if not items:
        return
    cols = 3
    rows = math.ceil(len(items) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 1.9 * rows + 0.3), squeeze=False)
    colors = {"door": ORANGE, "window": AQUA, "gap": INK2}
    for ax, (wid, E, ops) in zip(axs.flat, items):
        L, nl = len(E) * px, E.shape[1]
        ax.imshow(E.T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1.6,
                  extent=[0, L, 0, nl * layer_m], interpolation="nearest")
        for o in ops:
            s0, s1 = o["s"]
            z0 = o["sill"] or 0.0
            z1 = o["head"] if o["head"] is not None else nl * layer_m
            ax.add_patch(Rectangle((s0, z0), s1 - s0, z1 - z0, fill=False, edgecolor=colors[o["type"]], lw=1.6))
        ax.set_title(f"{wid}  {L:.2f} м".replace(".", ","), loc="left", fontsize=9, color=INK)
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(labelsize=7, colors=INK2)
    for ax in list(axs.flat)[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------

def floorplan(path, out_dir, px: float = 0.01, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels, angle, rasters = build_rasters(path, px=px, log=log)
    result = {"file": Path(path).name, "rotation_deg": round(angle, 3),
              "transform": "plan = Rz(-rotation_deg) * world (x, y); z - над полом уровня",
              "levels": []}
    for k, (lv, ras) in enumerate(zip(levels, rasters), 1):
        plan, masks = analyse_level(ras, lv, k)
        name = "plan.png" if len(levels) == 1 else f"plan_L{k}.png"
        render(plan, ras, masks, out / name, title=f"Уровень {k}" if len(levels) > 1 else "")
        render_elevations(masks["elevations"], ras.px, ras.layer_m,
                          out / ("walls.png" if len(levels) == 1 else f"walls_L{k}.png"))
        d = asdict(plan)
        result["levels"].append(d)
        log(f"уровень {k}: стен {len(plan.walls)}, проёмов {len(plan.openings)}, помещений {len(plan.rooms)}, "
            f"{sum(r.area for r in plan.rooms):.1f} м²")
    (out / "elements.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    return result
