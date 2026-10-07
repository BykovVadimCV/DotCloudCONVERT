"""Вход сети из облака: псевдочертёж по свободному пространству (шаг 2 конвертера, раздел 5.5 ТЗ).

U-Net ReFloorBRUSNIKA обучена на чертежах: чёрные стены на белом. Облако приводится к такому
же виду без нейросети, только по лучам сканера:

  1. облако E57 переводится в СК планировки; пол и потолок каждой станции - пики гистограммы z;
  2. на станцию - полярные корзины по 0,5 градуса. Дальность корзины - k-я по величине
     горизонтальная дальность точек пола и потолка (k-я, а не максимум: одиночные выбросы,
     фасады за окном). Пол заслоняет мебель, потолок - почти никогда, поэтому берутся оба;
     но пол не дальше потолка + 0,6 м: за окном бывает земля на уровне пола, потолка там нет.
     Затем граница дотягивается до точек стены в пределах 0,25 м (пол у стены редкий).
     Пустые корзины до 3 градусов заполняются по соседям; многоугольник с отступом 2 см;
  3. объединение по станциям = свободное пространство F;
  4. здание B = закрытие F ядром 0,6 м (перегородки и несущие стены тоньше 0,6 м заливаются).
     Внутренние стены = B без F. Оставшиеся дыры шире 0,6 м - не стены, а то, куда лучи
     не попали (тень за углом, неотсканированная комната): «неизвестно». Наружные стены - кольцо 0,4 м вокруг B (за наружную стену сканер не видит,
     толщина принимается по умолчанию, как в ТЗ);
  5. признак окна: в корзине есть точки дальше границы (улица, соседний дом за стеклом).

Всё рисуется в кадре эталона U-Net (gt.json -> unet.frame), поэтому вход и маска совпадают
пиксель в пиксель. В конвертере кадр будет свой (оси по главным направлениям стен,
масштаб - стена 30 px), но вид входа тот же.

    python -m simscan pair D:/synth/scene_00000   ->  input/free_space.png, valid.png, pair.png

input/valid.png - где скан что-то видел (здание из свободного пространства + кольцо + 0,3 м).
Вне её эталон есть, а данных нет (неотсканированная комната) - это маска для функции потерь.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .groundtruth import RasterFrame
from .transform import WorldTransform

INPUT_VALUES = {"free": 255, "outside": 255, "unknown": 215, "window": 140, "wall": 0}


def _disk(radius_px: float):
    import cv2

    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _plane_z(rel: np.ndarray, lo: float, hi: float) -> float | None:
    """Пик гистограммы высот (шаг 1 см) в окне [lo, hi] относительно станции."""
    m = (rel > lo) & (rel < hi)
    if m.sum() < 100:
        return None
    h, e = np.histogram(rel[m], bins=np.arange(lo, hi + 0.01, 0.01))
    k = int(h.argmax())
    return float((e[k] + e[k + 1]) / 2)


def station_polygon(c, pts: np.ndarray, floor_z: float, ceil_z: float | None, n_bins: int = 720,
                    k: int = 3, band: float = 0.04, inset: float = 0.02, max_gap: int = 6,
                    beyond: float = 0.3, ceil_margin: float = 0.6, wall_reach: float = 0.25):
    """Дальности корзин (n_bins,) и флаги «есть точки за границей» (n_bins,)."""
    d = pts[:, :2] - np.asarray(c[:2])
    r = np.hypot(d[:, 0], d[:, 1])
    b = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins

    def kth(sel):
        rb = np.zeros(n_bins)
        cnt = np.bincount(b[sel], minlength=n_bins)
        if sel.any():
            order = np.lexsort((r[sel], b[sel]))
            ends = np.cumsum(cnt)
            ok = cnt >= k
            rb[ok] = r[sel][order][ends[ok] - k]
        return rb

    r_floor = kth(np.abs(pts[:, 2] - floor_z) < band)
    if ceil_z is None:
        rb = r_floor
    else:
        # пол дальше потолка более чем на ceil_margin - это земля за окном, а не комната
        # (снаружи нет потолка); запас - на короба и подвесные потолки у стен
        r_ceil = kth(np.abs(pts[:, 2] - ceil_z) < band)
        rb = np.where(r_ceil > 0, np.maximum(r_ceil, np.minimum(r_floor, r_ceil + ceil_margin)),
                      r_floor)
    # дотянуть до стены: при крупном шаге точки пола у стены редкие (на 4 м при шаге 0,5° -
    # через 10 см), а стена рядом с границей плотная. Дальше wall_reach от границы не берём
    reach = (rb[b] > 0) & (r <= rb[b] + wall_reach)
    rb = np.maximum(rb, kth(reach))
    # короткие пропуски - по соседям (минимум из двух сторон: осторожно)
    empty = rb <= 0
    if empty.any() and not empty.all():
        idx = np.arange(n_bins)
        filled = rb.copy()
        for i in idx[empty]:
            left = right = None
            for s in range(1, max_gap + 1):
                if left is None and rb[(i - s) % n_bins] > 0:
                    left = rb[(i - s) % n_bins]
                if right is None and rb[(i + s) % n_bins] > 0:
                    right = rb[(i + s) % n_bins]
            if left is not None and right is not None:
                filled[i] = min(left, right)
        rb = filled
    rb = np.maximum(rb - inset, 0)
    # точки дальше границы: стекло, открытое окно, зеркало (мнимые точки)
    far = r > rb[b] + beyond
    beyond_cnt = np.bincount(b[far], minlength=n_bins)
    return rb, beyond_cnt >= k


def _sector_poly(frame: RasterFrame, c, r0, r1, n_bins: int, bins) -> list:
    """Многоугольники секторов-колец [r0, r1] по корзинам (в пикселях, для cv2.fillPoly)."""
    polys = []
    step = 2 * math.pi / n_bins
    for i in bins:
        a0, a1 = -math.pi + i * step, -math.pi + (i + 1) * step
        pts = [(c[0] + r0[i] * math.cos(a0), c[1] + r0[i] * math.sin(a0)),
               (c[0] + r1[i] * math.cos(a0), c[1] + r1[i] * math.sin(a0)),
               (c[0] + r1[i] * math.cos(a1), c[1] + r1[i] * math.sin(a1)),
               (c[0] + r0[i] * math.cos(a1), c[1] + r0[i] * math.sin(a1))]
        ii, jj = frame.xy_to_ij(*np.array(pts).T)
        polys.append(np.round(np.c_[jj, ii] * 16).astype(np.int32))
    return polys


def free_space_input(stations: list[dict], frame: RasterFrame, n_bins: int = 720,
                     close_m: float = 0.6, ext_wall_m: float = 0.4) -> dict:
    """stations: [{"center": (x, y, z), "points": (n, 3)}] в СК планировки."""
    import cv2

    H, W = frame.height, frame.width
    px = frame.pixel_m
    free = np.zeros((H, W), np.uint8)
    window = np.zeros((H, W), np.uint8)
    info = []
    for st in stations:
        c, p = np.asarray(st["center"], float), st["points"]
        rel = p[:, 2] - c[2]
        fz, cz = _plane_z(rel, -2.5, -0.3), _plane_z(rel, 0.3, 4.0)
        if fz is None:
            continue
        rb, far = station_polygon(c, p, c[2] + fz, None if cz is None else c[2] + cz, n_bins)
        step = 2 * math.pi / n_bins
        ang = -math.pi + np.repeat(np.arange(n_bins), 2) * step + np.tile([0, step], n_bins)
        rr = np.repeat(rb, 2)
        ii, jj = frame.xy_to_ij(c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang))
        cv2.fillPoly(free, [np.round(np.c_[jj, ii] * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        if far.any():
            cv2.fillPoly(window, _sector_poly(frame, c, rb, rb + ext_wall_m + 0.05, n_bins,
                                              np.flatnonzero(far)), 1, cv2.LINE_8, 4)
        info.append({"floor_rel_m": round(fz, 3), "ceiling_rel_m": None if cz is None else round(cz, 3),
                     "bins_empty": int((rb <= 0).sum())})
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(0.03 / px))      # щели между секторами
    building = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(close_m / 2 / px))
    # дыры, которые закрытие не залило (шире close_m) - «неизвестно»
    n, lab, stats, _ = cv2.connectedComponentsWithStats((building == 0).astype(np.uint8), 4)
    unknown = np.zeros_like(building)
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):            # не снаружи здания
            unknown[lab == k] = 1
    building = building | unknown
    ring = cv2.dilate(building, _disk(ext_wall_m / px)) & (1 - building)
    walls = (building & (1 - free) & (1 - unknown)) | ring
    img = np.full((H, W), INPUT_VALUES["outside"], np.uint8)
    img[unknown.astype(bool)] = INPUT_VALUES["unknown"]
    img[walls.astype(bool)] = INPUT_VALUES["wall"]
    win = window.astype(bool) & walls.astype(bool)
    img[win] = INPUT_VALUES["window"]
    # зона, где скан что-то видел: здание с кольцом + 0,3 м. За ней эталон не проверяем
    # (неотсканированные комнаты), при обучении - маска потерь
    valid = cv2.dilate(building | ring, _disk(0.3 / px)) & (1 - unknown)
    return {"image": img, "free": free.astype(bool), "walls": walls.astype(bool),
            "window": win, "unknown": unknown.astype(bool), "valid": valid.astype(bool),
            "stations": info}


def load_stations(scene_dir: Path, world: WorldTransform, max_points: int = 3_000_000) -> list[dict]:
    from .e57io import read_e57_world

    out = []
    for s in read_e57_world(scene_dir / "scan.e57"):
        xyz = s["xyz"]
        c = world.to_layout(np.asarray(s["position"], float)[None])[0]
        p = world.to_layout(xyz[np.isfinite(xyz).all(1)])
        p = p[np.linalg.norm(p - c, axis=1) > 0.05]                     # недействительные - в нуле
        if len(p) > max_points:
            p = p[np.random.default_rng(0).choice(len(p), max_points, replace=False)]
        out.append({"center": c, "points": p})
    return out


def wall_scores(pred_walls: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None) -> dict:
    gt = (mask == 64) | (mask == 128)
    out = {}
    for suffix, m in (("", None), ("_scanned", valid)):
        if suffix and m is None:
            continue
        p, g = (pred_walls, gt) if m is None else (pred_walls & m, gt & m)
        inter, union = (p & g).sum(), (p | g).sum()
        out.update({f"wall_iou{suffix}": round(float(inter / max(union, 1)), 4),
                    f"wall_precision{suffix}": round(float(inter / max(p.sum(), 1)), 4),
                    f"wall_recall{suffix}": round(float(inter / max(g.sum(), 1)), 4)})
    return out


def make_pair(scene_dir, figure: bool = True) -> dict:
    import cv2

    scene_dir = Path(scene_dir)
    info = json.loads((scene_dir / "gt" / "gt.json").read_text(encoding="utf-8"))
    if "unet" not in info:
        raise ValueError("в сцене нет эталона U-Net (export.unet_mask)")
    frame = RasterFrame(**info["unet"]["frame"])
    w = info["layout_to_world"]
    world = WorldTransform(w["yaw_deg"], tuple(w["offset_m"]))
    stations = load_stations(scene_dir, world)
    res = free_space_input(stations, frame)
    out = scene_dir / "input"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "free_space.png"), cv2.merge([res["image"]] * 3))
    cv2.imwrite(str(out / "valid.png"), res["valid"].astype(np.uint8) * 255)
    mask = cv2.imread(str(scene_dir / "gt" / info["unet"]["file"]), cv2.IMREAD_GRAYSCALE)
    stats = {"frame_px": [frame.height, frame.width], "pixel_mm": round(frame.pixel_m * 1000, 2),
             "stations": res["stations"], **wall_scores(res["walls"], mask, res["valid"])}
    if figure:
        pair_figure(stations, frame, res, mask, stats, out / "pair.png")
    (out / "input.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return stats


def pair_figure(stations, frame: RasterFrame, res: dict, mask: np.ndarray, stats: dict,
                path: Path) -> None:
    import cv2

    from .debug import AQUA, BLUE, INK2, ORANGE, SURFACE, _plt

    plt = _plt()
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    H, W = mask.shape
    # срез облака 1,0-1,6 м над полом станции - то, из чего «на глаз» видно стены
    dens = np.zeros((H, W))
    for st in stations:
        c, p = np.asarray(st["center"]), st["points"]
        fz = _plane_z(p[:, 2] - c[2], -2.5, -0.3)
        if fz is None:
            continue
        z = p[:, 2] - (c[2] + fz)
        q = p[(z > 1.0) & (z < 1.6)]
        i, j = frame.xy_to_ij(q[:, 0], q[:, 1])
        i, j = np.round(i).astype(int), np.round(j).astype(int)
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        np.add.at(dens, (i[ok], j[ok]), 1)
    dens = cv2.dilate(np.log1p(dens), np.ones((3, 3)))                 # тонкие стены видны и в уменьшении
    dens = 1 - np.clip(dens / max(np.percentile(dens[dens > 0], 95), 1e-9), 0, 1)

    def rgb(c):
        return np.array(to_rgb(c))

    target = np.ones((H, W, 3)) * rgb(SURFACE)
    target[mask == 64] = rgb("#0b0b0b")
    target[mask == 128] = rgb(AQUA)
    target[mask == 192] = rgb(ORANGE)

    gt = (mask == 64) | (mask == 128)
    pw = res["walls"]
    over = np.ones((H, W, 3)) * rgb(SURFACE)
    over[pw & gt] = rgb("#9a9893")
    over[pw & ~gt] = rgb(ORANGE)
    over[gt & ~pw] = rgb(BLUE)
    over[~res["valid"]] = 0.55 * over[~res["valid"]] + 0.45 * rgb("#d6d4cf")    # вне скана - притушено

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.4), constrained_layout=True)
    panels = [
        (dens, "gray", "1. Облако: срез 1,0-1,6 м над полом"),
        (res["image"], "gray", "2. Вход сети: псевдочертёж\nпо свободному пространству"),
        (target, None, "3. Эталон: маска U-Net\n(стена 64, окно 128, дверь 192)"),
        (over, None, f"4. Вход против эталона: IoU стен {stats['wall_iou']:.2f}\n"
                     f"(в отсканированной зоне {stats['wall_iou_scanned']:.2f}; вне её - притушено)"),
    ]
    for st in stations:
        i, j = frame.xy_to_ij(st["center"][0], st["center"][1])
        axes[0].plot(j, i, marker="o", ms=7, color=ORANGE, mec="white", mew=1.2)
    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if img.dtype == np.uint8 else 1,
                  interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d6d4cf")
    axes[1].legend(handles=[Patch(color="black", label="стена (за кольцом - 0,4 м по умолчанию)"),
                            Patch(color="#8c8c8c", label="признак окна (лучи ушли наружу)"),
                            Patch(facecolor="#d7d7d7", edgecolor="#bbb", label="не отсканировано")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=9, frameon=False)
    axes[2].legend(handles=[Patch(color="#0b0b0b", label="стена"), Patch(color=AQUA, label="окно"),
                            Patch(color=ORANGE, label="дверь")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
    axes[3].legend(handles=[Patch(color="#9a9893", label="совпало"),
                            Patch(color=ORANGE, label="лишняя стена во входе"),
                            Patch(color=BLUE, label="стена пропущена")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Обучающая пара (оранжевые точки - станции): {frame.width}x{frame.height} px, "
                 f"пиксель {stats['pixel_mm']} мм.  Вход без сети, отсканированная зона: точность стен "
                 f"{stats['wall_precision_scanned']:.2f}, полнота {stats['wall_recall_scanned']:.2f}",
                 fontweight="bold", color=INK2)
    fig.savefig(path, dpi=80)
    plt.close(fig)
