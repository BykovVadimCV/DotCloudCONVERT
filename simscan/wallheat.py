"""Тепловая карта стен: наложение горизонтальных срезов облака с весами.

    python -m simscan wall-heatmap D:/scans/kvartira.e57 --out D:/heat
    -> heat.png (на уровень - heat_L1.png ...), slices.png, heat.npy, weights.json

Срез - точки в слое высоты (по умолчанию 10 см) над полом уровня, растр занятости. Стена
вертикальна: она есть почти во всех срезах на одном и том же месте. Шум и бред - нет: мебель
и радиаторы - в нескольких нижних срезах, светильники и короба - в верхних, пол и потолок
заливают срез целиком, окна и двери выпадают из средних срезов.

Вес среза - насколько его картина совпадает с устойчивой по высоте структурой:
  точность - доля занятого в срезе, что лежит на структуре (мало - шум, заливка);
  полнота  - доля структуры, что видна в срезе (мало - окна, проёмы, тень);
  качество = F1 (гармоническое среднее);
вес = априорный вес по высоте x качество. По высоте (по умолчанию): у пола 1 - там стена
сплошная, середина 1,0-2,0 м - 0,1 - там окна, верх - 0,6. Задаётся --prior.
Структура - пиксели, занятые хотя бы в 4 срезах (40 см
по высоте; с весами - на каждом проходе заново). Не «в половине срезов»: иначе стена под
окном и над ним выпадает из структуры и средние срезы, где окна, ничего не теряют.

Карта = сумма весов срезов, где пиксель занят, / сумма весов: 1 - стена во всю высоту,
~0,6 - стена с окном, ~0,3 - перемычка над дверью, около 0 - шум.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _frame(path, px: float, chunk: int, sample_points: int, log):
    """Уровни, поворот вдоль стен и кадр растра - по выборке точек."""
    from .e57read import E57Reader
    from .floorplan import _scan_points, dominant_angle
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
    lo = np.percentile(q, 0.2, axis=0) - 0.5
    hi = np.percentile(q, 99.8, axis=0) + 0.5
    px = max(px, float((hi - lo).max()) / 3000)
    shape = (int(math.ceil((hi[1] - lo[1]) / px)), int(math.ceil((hi[0] - lo[0]) / px)))
    log(f"уровней {len(levels)}, поворот {angle:.2f}°, растр {shape[1]}x{shape[0]} по {px * 1000:.0f} мм")
    return levels, angle, rot, lo, px, shape


def slice_counts(path, px: float = 0.01, slice_m: float = 0.1, margin_m: float = 0.1,
                 chunk: int = 2_000_000, sample_points: int = 3_000_000, log=print):
    """Второй проход: на каждый уровень - счётчики точек (срез, H, W), uint8 с насыщением.
    Срезы от margin_m над полом до margin_m под потолком."""
    from .e57read import E57Reader
    from .floorplan import _scan_points

    levels, angle, rot, lo, px, (H, W) = _frame(path, px, chunk, sample_points, log)
    out = []
    for lv in levels:
        top = lv["ceiling_z"] - lv["floor_z"]
        edges = np.arange(margin_m, top - margin_m + 1e-9, slice_m)
        out.append({"level": lv, "z0": edges[:-1], "slice_m": slice_m,
                    "counts": np.zeros((len(edges) - 1, H, W), np.uint8)})
    with E57Reader(path) as r:
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                q = p[:, :2] @ rot.T
                jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
                flat = ii * W + jj
                for L in out:
                    if not len(L["z0"]):
                        continue
                    h = p[:, 2] - L["level"]["floor_z"]
                    k = np.floor((h - L["z0"][0]) / slice_m).astype(np.int64)
                    m = ok & (k >= 0) & (k < len(L["z0"]))
                    if not m.any():
                        continue
                    key = k[m] * (H * W) + flat[m]
                    u, cnt = np.unique(key, return_counts=True)
                    c = L["counts"].reshape(-1)
                    c[u] = np.minimum(c[u].astype(np.int32) + cnt, 255).astype(np.uint8)
    return out, {"rotation_deg": angle, "origin": lo.tolist(), "px": px, "shape": [H, W]}


DEFAULT_PRIOR = "0:1,0.8:1,1.0:0.1,2.0:0.1,2.2:0.6,9:0.6"


def height_prior(z: np.ndarray, spec: str = DEFAULT_PRIOR) -> np.ndarray:
    """Априорный вес среза по высоте над полом: ломаная «высота:вес,...». По умолчанию низ
    (до 0,8 м) - 1: там стена сплошная; середина 1,0-2,0 м - 0,1: там окна; верх - 0,6:
    перемычки есть, но и короба, светильники."""
    pts = sorted((float(a), float(b)) for a, b in (t.split(":") for t in spec.split(",")))
    return np.interp(z, [p[0] for p in pts], [p[1] for p in pts])


def slice_weights(occ: np.ndarray, prior: np.ndarray | None = None, iters: int = 3, tol_px: int = 1,
                  min_slices: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """occ (срез, H, W) bool -> (веса срезов, карта, разбор: точность, полнота)."""
    import cv2

    n = len(occ)
    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    dil = np.stack([cv2.dilate(o.astype(np.uint8), k) for o in occ]).astype(bool)
    prior = np.ones(n) if prior is None else np.asarray(prior, float)
    w = prior.copy()
    f1 = np.ones(n)
    support = np.tensordot(w, occ.astype(np.float32), 1)   # во скольких срезах занят (с весами)
    prec = rec = np.zeros(n)
    for _ in range(iters):
        # структура - занято хотя бы в min_slices срезах (40 см по высоте), а не в половине:
        # стена под окном и над ним тоже структура, срезы через окна её не видят - полнота ниже
        ref = support >= min_slices * w.mean()
        ref_d = cv2.dilate(ref.astype(np.uint8), k).astype(bool)
        nr = max(int(ref.sum()), 1)
        prec = np.array([(o & ref_d).sum() / max(int(o.sum()), 1) for o in occ])
        rec = np.array([(ref & d).sum() / nr for d in dil])
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
        w = prior * f1                               # вес = высота (априори) x качество среза
        support = np.tensordot(w, occ.astype(np.float32), 1)
    heat = support / max(w.sum(), 1e-9)
    return w, heat, {"precision": prec, "recall": rec, "f1": f1, "prior": prior}


def wall_heatmap(path, out_dir, px: float = 0.01, slice_m: float = 0.1, prior: str = DEFAULT_PRIOR,
                 log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, frame = slice_counts(path, px=px, slice_m=slice_m, log=log)
    result = {"file": Path(path).name, **frame, "prior": prior, "levels": []}
    for li, L in enumerate(data, 1):
        occ = L["counts"] > 0
        w, heat, info = slice_weights(occ, height_prior(L["z0"] + L["slice_m"] / 2, prior))
        sfx = "" if len(data) == 1 else f"_L{li}"
        np.save(out / f"heat{sfx}.npy", heat.astype(np.float16))
        _plot_heat(heat, L, w, info, frame["px"], out / f"heat{sfx}.png")
        _plot_slices(occ, L, w, out / f"slices{sfx}.png")
        result["levels"].append({
            "floor_z": round(L["level"]["floor_z"], 3), "ceiling_z": round(L["level"]["ceiling_z"], 3),
            "slices": [{"z": round(float(z), 2), "weight": round(float(a), 3), "prior": round(float(pr), 3),
                        "f1": round(float(f), 3), "precision": round(float(p), 3), "recall": round(float(r), 3)}
                       for z, a, pr, f, p, r in zip(L["z0"], w, info["prior"], info["f1"], info["precision"],
                                                    info["recall"])]})
        log(f"уровень {li}: срезов {len(w)}, вес > 0,5 у {int((w > 0.5).sum())}")
    (out / "weights.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


# ----------------------------------------------------------------------

def _crop(heat: np.ndarray, pad: int = 20):
    ys, xs = np.nonzero(heat > 0.05)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(0, ys.min() - pad), ys.max() + pad), slice(max(0, xs.min() - pad), xs.max() + pad))


def _plot_heat(heat, L, w, info, px, path):
    from .debug import INK, INK2, ORANGE, _plt

    plt = _plt()
    sl = _crop(heat)
    h = heat[sl]
    H, W = h.shape
    fig = plt.figure(figsize=(12 * W / max(H, W) + 3.6, 12 * H / max(H, W) + 0.6))
    ax = fig.add_axes([0.01, 0.03, 0.74, 0.92])
    im = ax.imshow(h, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    ax.plot([20, 20 + 1 / px], [H - 20] * 2, color=INK, lw=2)
    ax.text(20 + 0.5 / px, H - 28, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("Стены: взвешенная доля срезов, где пиксель занят", loc="left", fontsize=10, color=INK)
    ax.axis("off")
    cax = fig.add_axes([0.03, 0.01, 0.25, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=7, colors=INK2)
    bx = fig.add_axes([0.80, 0.14, 0.17, 0.79])
    z = L["z0"] + L["slice_m"] / 2
    bx.barh(z, w, height=L["slice_m"] * 0.8, color="#3b6fb6", label="вес")
    bx.plot(info["prior"], z, color=INK, lw=1.2, ls=":", label="по высоте")
    bx.plot(info["f1"], z, color=ORANGE, lw=1.4, label="качество (F1)")
    bx.plot(info["precision"], z, color=ORANGE, lw=0.7, alpha=0.6, label="точность")
    bx.plot(info["recall"], z, color=INK2, lw=1, ls="--", label="полнота")
    bx.set_xlim(0, 1)
    bx.set_ylabel("высота над полом, м", fontsize=8, color=INK2)
    bx.set_title("Вес среза", loc="left", fontsize=10, color=INK)
    bx.tick_params(labelsize=7, colors=INK2)
    bx.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.04), ncol=2)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_slices(occ, L, w, path, cols: int = 6):
    from .debug import INK, _plt

    plt = _plt()
    sl = _crop(occ.mean(0))
    n = len(occ)
    rows = math.ceil(n / cols)
    H, W = occ[0][sl].shape
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6 * H / max(W, 1) + 0.3), squeeze=False)
    for ax, k in zip(axs.flat, range(n)):
        ax.imshow(occ[k][sl], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        z = L["z0"][k]
        ax.set_title(f"{z:.1f}-{z + L['slice_m']:.1f} м  вес {w[k]:.2f}".replace(".", ","), loc="left",
                     fontsize=7, color=INK)
        ax.axis("off")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
