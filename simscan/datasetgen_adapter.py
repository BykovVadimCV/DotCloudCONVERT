"""Планировки из datasetgen (ReFloorBRUSNIKA) -> Layout simscan.

datasetgen живёт в пикселях и рисует двери и окна сразу в растр: в структурах
данных есть только стены (оси, толщина в px, наружная / ограждение / стена балкона),
прямоугольники комнат-меток и балконы. Поэтому:

  1. extract_plan() повторяет геометрическую часть generate_single_plan
     (стратегия + балконы) и отдаёт словарь «plan» - его же можно сохранить
     в JSON на стороне datasetgen и подать сюда без импорта модуля;
  2. layout_from_plan() переводит px в метры (длинная сторона здания
     разыгрывается в метрах), толщины стен разыгрывает заново в метрах
     (в px они от 3 до 200 и физического смысла не имеют, но тонкие перегородки
     остаются тонкими), помещения строит как связные области между стенами
     (у части стратегий прямоугольники комнат не покрывают план или
     перекрываются), а двери и окна ставит сам - уже с высотами.

Стены - первичная геометрия: прямоугольники комнат используются только для
типа помещения (кухня, санузел, коридор, балкон).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import LayoutConfig
from .layout import Layout, LayoutGenerator, Opening, Room, Wall

DEFAULT_STRATEGIES = ("central", "linear", "radial", "graph", "open_plan",
                      "asymmetric", "courtyard", "spine")
ROOM_KIND = {"bathroom": "bath", "laundry": "bath", "kitchen": "kitchen",
             "corridor": "corridor", "walk_in_closet": "corridor"}

OUTSIDE, SOLID = -2, -1


# ----------------------------------------------------------------------
# 1. Извлечение геометрии из datasetgen
# ----------------------------------------------------------------------

def load_datasetgen(path: str | Path):
    """Импорт datasetgen.py по пути к файлу или к каталогу с ним."""
    path = Path(path)
    folder = path.parent if path.suffix == ".py" else path
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("datasetgen")


def extract_plan(dg, seed: int, strategies=DEFAULT_STRATEGIES,
                 balcony_probability: float | None = None) -> dict:
    """Геометрия одного плана datasetgen (без рендера) в виде словаря."""
    kwargs = {"seed": seed, "strategies": list(strategies)}
    if balcony_probability is not None:
        kwargs["balcony_probability"] = balcony_probability
    cfg = dg.FloorPlanConfig(**kwargs)
    gen = dg.SmartFloorPlanGenerator(cfg)
    rng = gen.rng
    # толщины в px - как в generate_dataset; нужны только их отношение
    ext_th = rng.randint(*cfg.wall_thickness_range) * gen.super_sampling
    if rng.random() < cfg.thin_partition_prob:
        ratio = rng.uniform(*cfg.thin_partition_ratio_range)
    else:
        ratio = rng.uniform(*cfg.int_wall_ratio_range)
    int_th = max(int(ext_th * ratio), cfg.int_wall_min_px * gen.super_sampling)

    margin = int(gen.size * rng.uniform(0.12, 0.28))
    bounds = (margin, margin, gen.size - margin, gen.size - margin)
    brief = gen.generate_brief()
    strategy = gen.strategies[brief.circulation_type]
    walls, rooms = strategy.generate(bounds, ext_th, int_th, gen.solver, brief, cfg)
    balconies = []
    if brief.has_balcony:
        walls, rooms, balconies = gen._integrate_balconies(walls, rooms, ext_th)
    return {
        "source": "datasetgen", "seed": seed, "strategy": brief.circulation_type,
        "size_px": gen.size, "bounds_px": list(bounds),
        "ext_th_px": ext_th, "int_th_px": int_th,
        "walls": [{"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2, "thickness": w.thickness,
                   "is_external": bool(w.is_external), "is_guardrail": bool(w.is_guardrail),
                   "is_balcony_interface": bool(w.is_balcony_interface)} for w in walls],
        "rooms": [{"bounds": [r.x1, r.y1, r.x2, r.y2], "type": r.room_type} for r in rooms],
        "balconies": [{"bounds": [min(b.x1, b.x2), min(b.y1, b.y2), max(b.x1, b.x2),
                                  max(b.y1, b.y2)], "side": b.side} for b in balconies],
    }


# ----------------------------------------------------------------------
# 2. Перевод в Layout
# ----------------------------------------------------------------------

class _Grid:
    """Сжатая сетка по граням стен: клетка - стена, улица или помещение."""

    def __init__(self, rects: list, pad: float = 1.0):
        xs = {r[0] for r in rects} | {r[2] for r in rects}
        ys = {r[1] for r in rects} | {r[3] for r in rects}
        self.xs = np.array(sorted(xs | {min(xs) - pad, max(xs) + pad}))
        self.ys = np.array(sorted(ys | {min(ys) - pad, max(ys) + pad}))
        ny, nx = len(self.ys) - 1, len(self.xs) - 1
        solid = np.zeros((ny, nx), bool)
        for x0, y0, x1, y1 in rects:
            i0, i1 = np.searchsorted(self.xs, [x0, x1])
            j0, j1 = np.searchsorted(self.ys, [y0, y1])
            solid[j0:j1, i0:i1] = True
        lab, n = ndimage.label(~solid)
        border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
        self.cls = np.full((ny, nx), SOLID, int)
        self.cls[np.isin(lab, list(border - {0}))] = OUTSIDE
        self.components = [np.argwhere(lab == k) for k in range(1, n + 1) if k not in border]
        self.solid = solid

    def cell_rect(self, j, i):
        return (float(self.xs[i]), float(self.ys[j]), float(self.xs[i + 1]), float(self.ys[j + 1]))

    def classify(self, x: float, y: float) -> int:
        i = int(np.searchsorted(self.xs, x, side="right")) - 1
        j = int(np.searchsorted(self.ys, y, side="right")) - 1
        if not (0 <= i < self.cls.shape[1] and 0 <= j < self.cls.shape[0]):
            return OUTSIDE
        return int(self.cls[j, i])


def _merge_cells(grid: _Grid, cells: np.ndarray) -> list:
    """Клетки компоненты -> покрытие прямоугольниками (полосы по строкам, затем по столбцам)."""
    rows: dict[int, list] = {}
    for j, i in sorted(map(tuple, cells)):
        runs = rows.setdefault(j, [])
        if runs and runs[-1][1] == i:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    rects = []
    open_: dict[tuple, list] = {}
    for j in sorted(rows):
        nxt = {}
        for i0, i1 in rows[j]:
            key = (i0, i1)
            if key in open_ and open_[key][3] == j:
                open_[key][3] = j + 1
                nxt[key] = open_.pop(key)
            else:
                nxt[key] = [i0, j, i1, j + 1]
        rects += list(open_.values())
        open_ = nxt
    rects += list(open_.values())
    return [(float(grid.xs[i0]), float(grid.ys[j0]), float(grid.xs[i1]), float(grid.ys[j1]))
            for i0, j0, i1, j1 in rects]


def _wall_edges(grid: _Grid, cells: np.ndarray) -> list:
    """Грани между клетками помещения и стенами, нормаль внутрь помещения."""
    own = {tuple(c) for c in cells}
    edges = []
    for j, i in own:
        x0, y0, x1, y1 = grid.cell_rect(j, i)
        for dj, di, edge, n in ((0, -1, (x0, y0, x0, y1), (1, 0)), (0, 1, (x1, y0, x1, y1), (-1, 0)),
                                (-1, 0, (x0, y0, x1, y0), (0, 1)), (1, 0, (x0, y1, x1, y1), (0, -1))):
            jj, ii = j + dj, i + di
            if (jj, ii) in own:
                continue
            if 0 <= jj < grid.solid.shape[0] and 0 <= ii < grid.solid.shape[1] and grid.solid[jj, ii]:
                edges.append((*edge, *n))
    # слить соседние коллинеарные
    edges.sort(key=lambda e: (e[4], e[5], e[0] if e[4] else e[1], e[1] if e[4] else e[0]))
    out = []
    for e in edges:
        if out:
            p = out[-1]
            same_line = p[4:] == e[4:] and ((e[4] and p[0] == e[0] and abs(p[3] - e[1]) < 1e-9)
                                            or (e[5] and p[1] == e[1] and abs(p[2] - e[0]) < 1e-9))
            if same_line:
                out[-1] = (p[0], p[1], e[2], e[3], p[4], p[5])
                continue
        out.append(e)
    return out


def layout_from_plan(plan: dict, cfg: LayoutConfig, rng: np.random.Generator) -> Layout:
    left, top, right, bottom = plan["bounds_px"]
    long_px = max(right - left, bottom - top)
    scale = float(rng.uniform(*cfg.datasetgen_long_side_m)) / long_px

    def xy(px, py):                      # px -> м, ось Y вверх
        return round((px - left) * scale, 4), round((bottom - py) * scale, 4)

    t_ext = float(rng.choice(cfg.exterior_wall_m))
    thin = plan["int_th_px"] < 0.4 * plan["ext_th_px"]
    pool = [t for t in cfg.interior_wall_m if (t <= 0.12 if thin else t >= 0.12)] or list(cfg.interior_wall_m)
    t_int = float(rng.choice(pool))
    t_par = cfg.parapet_thickness_m
    H = round(float(rng.uniform(*cfg.ceiling_height_m)), 3)
    W, D = (right - left) * scale, (bottom - top) * scale

    # --- стены -----------------------------------------------------------
    walls: list[Wall] = []
    seen = set()
    for w in plan["walls"]:
        a, b = xy(w["x1"], w["y1"]), xy(w["x2"], w["y2"])
        if a == b:
            continue
        a, b = (a, b) if a <= b else (b, a)
        if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
            continue                     # наклонных стен в datasetgen нет; на всякий случай
        if w["is_guardrail"]:
            kind, t, height = "parapet", t_par, round(float(rng.uniform(*cfg.parapet_height_m)), 3)
        elif w["is_external"]:
            kind, t, height = "exterior", t_ext, None
        elif w["is_balcony_interface"]:
            kind, t, height = "interior", t_ext, None
        else:
            kind, t, height = "interior", t_int, None
        key = (a, b, kind)
        if key in seen:
            continue
        seen.add(key)
        wall = Wall(len(walls), a, b, t, kind, height=height,
                    role="balcony_interface" if w["is_balcony_interface"] else "")
        # наружные стены лежат осью на контуре: углы замыкаем удлинением на t/2
        for end, p in ((0, a), (1, b)):
            at_corner = (abs(p[0]) < 1e-6 or abs(p[0] - W) < 1e-6) and \
                        (abs(p[1]) < 1e-6 or abs(p[1] - D) < 1e-6)
            ext = t / 2 if (kind != "interior" and at_corner) else 0.01
            if end == 0:
                wall.ext0 = ext
            else:
                wall.ext1 = ext
        walls.append(wall)

    def body(w: Wall):
        pts = np.array([w.point(s, o) for s in (-w.ext0, w.length + w.ext1)
                        for o in (-w.thickness / 2, w.thickness / 2)])
        return (*pts.min(0).round(6), *pts.max(0).round(6))

    grid = _Grid([body(w) for w in walls])

    # --- помещения: связные области между стенами ------------------------
    rooms: list[Room] = []
    for cells in grid.components:
        region = _merge_cells(grid, cells)
        area = sum((c - a) * (d - b) for a, b, c, d in region)
        if area < cfg.datasetgen_min_room_m2:          # щели между стенами - пустоты
            continue
        rid = len(rooms)
        for j, i in cells:
            grid.cls[j, i] = rid
        arr = np.array(region)
        bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 2].max()), float(arr[:, 3].max()))
        rooms.append(Room(rid, bbox, bbox, "room", H, region=region,
                          wall_edges=_wall_edges(grid, cells)))
    if not rooms:
        raise ValueError("в плане datasetgen нет замкнутых помещений")
    _assign_kinds(plan, rooms, xy, cfg, rng)

    layout = Layout(H, rooms, walls, footprint=_merge_cells(grid, np.argwhere(grid.cls != OUTSIDE)),
                    meta={"source": "datasetgen", "datasetgen": {
                        k: plan[k] for k in ("seed", "strategy", "size_px", "bounds_px",
                                             "ext_th_px", "int_th_px")},
                        "scale_m_per_px": scale, "footprint_m": [W, D], "t_ext": t_ext,
                        "t_int": t_int})

    # --- проёмы ----------------------------------------------------------
    int_spans, ext_spans, balcony_spans = _spans(layout, grid)
    gen = LayoutGenerator(cfg, rng)
    gen._place_doors(layout, int_spans)
    _place_balcony_blocks(layout, balcony_spans, cfg, rng)
    ext_spans = [sp for sp in ext_spans if layout.rooms[sp[3]].kind != "balcony"]
    gen._place_entrance(layout, ext_spans)
    gen._place_windows(layout, ext_spans)
    return layout


def _assign_kinds(plan, rooms, xy, cfg, rng) -> None:
    """Тип помещения - по прямоугольнику-метке datasetgen с наибольшим перекрытием."""
    def overlap(r, rect):
        x0, y0, x1, y1 = rect
        return sum(max(0.0, min(c, x1) - max(a, x0)) * max(0.0, min(d, y1) - max(b, y0))
                   for a, b, c, d in r.rects())

    def to_m(bounds):
        (ax, ay), (bx, by) = xy(bounds[0], bounds[1]), xy(bounds[2], bounds[3])
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    labels = [(to_m(r["bounds"]), r["type"]) for r in plan["rooms"]]
    balconies = [to_m(b["bounds"]) for b in plan["balconies"]]
    for room in rooms:
        if any(overlap(room, b) > 0.5 * room.area for b in balconies):
            room.kind = "balcony"
            continue
        best = max(labels, key=lambda lt: overlap(room, lt[0]), default=None)
        if best and overlap(room, best[0]) > 0:
            room.kind = ROOM_KIND.get(best[1], "room")
        if rng.random() < cfg.p_suspended_ceiling and room.kind != "balcony":
            room.ceiling_z = round(room.ceiling_z - float(rng.uniform(*cfg.suspended_drop_m)), 3)


def _spans(layout: Layout, grid: _Grid):
    """Участки стен по сторонам: что за гранью - помещение, улица или другая стена."""
    int_spans, ext_spans, balcony_spans = [], [], []
    rooms = layout.rooms
    for w in layout.walls:
        if w.kind == "parapet":
            continue
        horizontal = abs(w.u[0]) > 0.5
        lo = w.p0[0] if horizontal else w.p0[1]
        brk = grid.xs if horizontal else grid.ys
        ss = sorted({0.0, w.length, *[float(v - lo) for v in brk if 0 < v - lo < w.length]})
        segs = []
        for a, b in zip(ss[:-1], ss[1:]):
            m = (a + b) / 2
            sides = tuple(grid.classify(*w.point(m, sd * (w.thickness / 2 + 1e-3))) for sd in (1, -1))
            if segs and segs[-1][2] == sides and abs(segs[-1][1] - a) < 1e-9:
                segs[-1][1] = b
            else:
                segs.append([a, b, sides])
        outside_votes = 0
        for a, b, (left, right) in segs:
            if left >= 0 and right >= 0 and left != right:
                pair = (min(left, right), max(left, right))
                if "balcony" not in (rooms[left].kind, rooms[right].kind):
                    int_spans.append((w.id, a, b, pair))
                elif w.role == "balcony_interface":     # боковые стены балкона - без проёмов
                    balcony_spans.append((w.id, a, b, pair))
            elif w.kind == "exterior" and (left == OUTSIDE) != (right == OUTSIDE):
                room = left if left >= 0 else right
                if room >= 0:
                    ext_spans.append((w.id, a, b, room))
                outside_votes += (b - a) * (1 if left == OUTSIDE else -1)
        if w.kind == "exterior":
            w.outward = 1 if outside_votes >= 0 else -1
        if w.role == "balcony_interface":
            for a, b, (left, right) in segs:
                if left >= 0 and rooms[left].kind == "balcony":
                    w.outward = 1
                elif right >= 0 and rooms[right].kind == "balcony":
                    w.outward = -1
    return int_spans, ext_spans, balcony_spans


def _place_balcony_blocks(layout: Layout, spans, cfg: LayoutConfig, rng) -> None:
    """Балконный блок: дверь в балкон и окно рядом на одной стене."""
    by_pair: dict = {}
    for wid, a, b, pair in spans:
        by_pair.setdefault(pair, []).append((wid, a, b))
    sill = round(float(rng.uniform(*cfg.window_sill_m)), 3)
    for pair, items in by_pair.items():
        wid, a, b = max(items, key=lambda x: x[2] - x[1])
        wall = layout.wall(wid)
        room = pair[0] if layout.rooms[pair[0]].kind != "balcony" else pair[1]
        dw = round(float(rng.uniform(0.7, 0.8)), 2)
        ww = round(float(rng.uniform(0.9, 1.5)), 2)
        m, gap = 0.25, 0.15
        if b - a < dw + 2 * m:
            continue
        head = round(min(float(rng.uniform(*cfg.door_head_m)), layout.ceiling_height - 0.1), 3)
        ww = min(ww, b - a - 2 * m - dw - gap)
        door_first = rng.random() < 0.5
        s_door = a + m + dw / 2 if door_first else b - m - dw / 2
        swing = 1 if layout.rooms[room].contains(*wall.point(s_door, wall.thickness / 2 + 0.02)) else -1
        layout.openings.append(Opening(len(layout.openings), wid, "door", round(s_door, 3), dw, 0.0,
                                       head, rooms=tuple(pair), hinge=int(rng.choice((-1, 1))),
                                       swing=swing))
        if ww >= 0.6:
            s_win = s_door + (dw / 2 + gap + ww / 2) * (1 if door_first else -1)
            layout.openings.append(Opening(len(layout.openings), wid, "window", round(s_win, 3), ww,
                                           sill, head, rooms=tuple(pair)))
