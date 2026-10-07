"""Анализ реального E57: всё, что нужно для калибровки синтетики и конвертера, в маленьком архиве.

    python -m simscan inspect-e57 D:/scans/kvartira.e57 --out D:/e57_report
    -> D:/e57_report.zip  (несколько МБ; его и пересылать)

Файл читается потоково, кусками по --chunk точек: память не зависит от размера файла
(на 20 ГБ - те же сотни МБ). Что извлекается:

  формат       версия, поля точек и их типы (float32/64, scaled integer и шаг), есть ли
               RGB / недействительные лучи / сетка row-column, встроенные снимки (только
               число и размер - сами снимки не копируются), байт на точку;
  станции      число точек, сетка строк и столбцов, шаг по азимуту и углу места, поза,
               наклон, высота над полом, пол и потолок, доля лучей без отклика;
  сенсор       распределения дальности и интенсивности; шум дальности по дальности и углу
               падения (отклонение точки от плоскости соседей по сетке); смешанные пиксели
               на перепадах (доля точек «между» поверхностями); скорость чтения и память;
  план         срез 1,0-1,6 м и вход сети (псевдочертёж по свободному пространству,
               rasterize.py) в СК файла;
  sample.npz   (по умолчанию; --no-sample - без него) прореженное облако: по станции не
               больше 200 тыс. точек в вокселях 4 см, xyz + интенсивность, без цвета.

Что в архив не попадает: цвет точек, фотографии, серийные номера (вычищаются из дерева
заголовка), полное облако.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

SUPPORTED = ("cartesianX", "cartesianY", "cartesianZ", "sphericalRange", "sphericalAzimuth",
             "sphericalElevation", "intensity", "rowIndex", "columnIndex", "cartesianInvalidState",
             "sphericalInvalidState")
RANGE_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 1e9)
INC_BINS = (0.0, 30.0, 50.0, 65.0, 75.0, 85.0)


# ----------------------------------------------------------------------
# Заголовок
# ----------------------------------------------------------------------

def _cast(node):
    from pye57 import libe57

    T = libe57.NodeType
    cast = {T.E57_BLOB: libe57.BlobNode, T.E57_COMPRESSED_VECTOR: libe57.CompressedVectorNode,
            T.E57_FLOAT: libe57.FloatNode, T.E57_INTEGER: libe57.IntegerNode,
            T.E57_SCALED_INTEGER: libe57.ScaledIntegerNode, T.E57_STRING: libe57.StringNode,
            T.E57_STRUCTURE: libe57.StructureNode, T.E57_VECTOR: libe57.VectorNode}
    if not isinstance(node, libe57.Node):               # уже типизированный узел
        return node
    return cast[node.type()](node)


def node_tree(node, depth: int = 0, max_children: int = 64):
    """Дерево заголовка E57 без данных точек и снимков; строки с «serial» вычищаются."""
    from pye57 import libe57

    n = _cast(node)
    name = node.elementName()
    if isinstance(n, libe57.StringNode):
        return "<скрыто>" if "serial" in name.lower() else n.value()
    if isinstance(n, (libe57.FloatNode, libe57.IntegerNode)):
        return n.value()
    if isinstance(n, libe57.ScaledIntegerNode):
        return {"scaled_integer": n.scaledValue(), "scale": n.scale(), "offset": n.offset(),
                "min": n.minimum(), "max": n.maximum()}
    if isinstance(n, libe57.BlobNode):
        return {"blob_bytes": int(n.byteCount())}
    if isinstance(n, libe57.CompressedVectorNode):
        return {"records": int(n.childCount())}                    # типы полей - field_types
    out = {} if isinstance(n, libe57.StructureNode) else []
    for i in range(min(n.childCount(), max_children)):
        c = n.get(i)
        v = node_tree(c, depth + 1)
        if isinstance(out, dict):
            out[c.elementName()] = v
        else:
            out.append(v)
    if n.childCount() > max_children:
        out = {"items": out, "truncated": int(n.childCount())} if isinstance(out, list) else out
    return out


def field_types(header) -> dict:
    from pye57 import libe57

    proto = libe57.StructureNode(header.points.prototype())
    out = {}
    for i in range(proto.childCount()):
        c = _cast(proto.get(i))
        name = proto.get(i).elementName()
        if isinstance(c, libe57.FloatNode):
            out[name] = "float64" if c.precision() == libe57.FloatPrecision.E57_DOUBLE else "float32"
        elif isinstance(c, libe57.ScaledIntegerNode):
            out[name] = f"scaled_integer(scale={c.scale():g}, bits~{_bits(c.minimum(), c.maximum())})"
        elif isinstance(c, libe57.IntegerNode):
            out[name] = f"integer[{c.minimum()}..{c.maximum()}]"
        else:
            out[name] = type(c).__name__
    return out


def _bits(lo, hi) -> int:
    return int(math.ceil(math.log2(max(hi - lo + 1, 2))))


def _blob_bytes(tree) -> int:
    if isinstance(tree, dict):
        return int(tree.get("blob_bytes", 0)) + sum(_blob_bytes(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_blob_bytes(v) for v in tree)
    return 0


# ----------------------------------------------------------------------
# Потоковое чтение
# ----------------------------------------------------------------------

def iter_chunks(e57, index: int, fields: list[str], chunk: int):
    """Куски станции; массивы - представления буфера, следующий кусок их перезапишет."""
    header = e57.get_header(index)
    data, buffers = e57.make_buffers(fields, chunk)
    reader = header.points.reader(buffers)
    try:
        while True:
            n = reader.read()
            if n == 0:
                break
            yield {k: v[:n] for k, v in data.items()}
    finally:
        reader.close()


def _quat_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ScanStats:
    """Накопители одной станции: всё в фиксированной памяти."""

    def __init__(self, n_points: int, rows: int | None, cols: int | None, keep_points: int,
                 keep_pairs: int, rng):
        self.rng = rng
        self.n = 0
        self.valid = 0
        self.p_keep = min(1.0, keep_points / max(n_points, 1))
        self.p_pair = min(1.0, 2 * keep_pairs / max(n_points, 1))
        self.keep_pairs = keep_pairs
        self.kept = []                                   # (xyz local float32, intensity)
        self.noise = []                                  # (range, incidence°, отклонение по нормали)
        self.jumps = [0, 0]                              # перепадов, из них «между»
        self.jump_pos = np.zeros(20, np.int64)
        self.az_step, self.el_step = [], []
        self.rows, self.cols = rows, cols
        self.pano = None
        if rows and cols:
            k = max(1, int(math.ceil(max(rows / 400, cols / 1600))))
            self.k = k
            shape = (rows // k + 1, cols // k + 1)
            self.pano = {"r": np.zeros(shape), "i": np.zeros(shape), "n": np.zeros(shape),
                         "all": np.zeros(shape)}

    def add(self, xyz: np.ndarray, inten, row, col, valid: np.ndarray) -> None:
        self.n += len(xyz)
        self.valid += int(valid.sum())
        if self.pano is not None and row is not None:
            P = self.pano
            shape = P["n"].shape
            flat = (row.astype(np.int64) // self.k) * shape[1] + col.astype(np.int64) // self.k
            size = shape[0] * shape[1]
            fv = flat[valid]
            xv = xyz[valid]
            r = np.sqrt(np.einsum("ij,ij->i", xv, xv))
            P["all"] += np.bincount(flat, minlength=size).reshape(shape)
            P["n"] += np.bincount(fv, minlength=size).reshape(shape)
            P["r"] += np.bincount(fv, weights=r, minlength=size).reshape(shape)
            if inten is not None:
                P["i"] += np.bincount(fv, weights=inten[valid], minlength=size).reshape(shape)
        keep = valid & (self.rng.random(len(xyz)) < self.p_keep)
        self.kept.append((xyz[keep].astype(np.float32),
                          None if inten is None else inten[keep].astype(np.float32)))
        if row is not None and col is not None:
            self._grid(xyz, row.astype(np.int64), col.astype(np.int64), valid)

    def _grid(self, xyz, row, col, valid) -> None:
        """Соседи по сетке внутри куска: шаг углов, шум относительно плоскости, перепады."""
        R = int(row.max()) + 2
        idx = np.flatnonzero(valid)
        if len(idx) < 100:
            return
        key = col[idx] * R + row[idx]
        order = np.argsort(key, kind="stable")
        ks, ids = key[order], idx[order]

        # центры - случайное подмножество: соседи ищутся по всем точкам куска, но считаем
        # только для центров (статистике хватает сотен тысяч, а время падает в разы)
        cen = np.flatnonzero(self.rng.random(len(ks)) < self.p_pair)
        ks_all = ks
        ks, ids = ks[cen], ids[cen]

        def nb(dk):                                      # сосед по всем точкам куска
            t = ks + dk
            j = np.clip(np.searchsorted(ks_all, t), 0, len(ks_all) - 1)
            return np.where(ks_all[j] == t, ids_all[j], -1)

        ids_all = idx[order]
        up, dn, lf, rt = nb(-1), nb(1), nb(-R), nb(R)
        p = xyz[ids]
        r = np.linalg.norm(p, axis=1)
        # шаг углов
        az = np.arctan2(p[:, 1], p[:, 0])
        el = np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))
        m = rt >= 0
        if m.any():
            pr = xyz[rt[m]]
            daz = np.abs(np.angle(np.exp(1j * (np.arctan2(pr[:, 1], pr[:, 0]) - az[m]))))
            self.az_step.append(np.median(np.degrees(daz)))
        m = dn >= 0
        if m.any():
            pd = xyz[dn[m]]
            del_ = np.abs(np.arctan2(pd[:, 2], np.hypot(pd[:, 0], pd[:, 1])) - el[m])
            self.el_step.append(np.median(np.degrees(del_)))
        # шум: отклонение точки от плоскости через 4 соседей, вдоль нормали
        m4 = (up >= 0) & (dn >= 0) & (lf >= 0) & (rt >= 0)
        sel = np.flatnonzero(m4)
        if len(sel):
            pu, pd, pl, pr = xyz[up[sel]], xyz[dn[sel]], xyz[lf[sel]], xyz[rt[sel]]
            c = p[sel]
            nrm = np.cross(pd - pu, pr - pl)
            nn = np.linalg.norm(nrm, axis=1)
            ok = nn > 1e-12
            nrm = nrm[ok] / nn[ok, None]
            mean4 = ((pu + pd + pl + pr) / 4)[ok]
            dev = np.abs(np.einsum("ij,ij->i", c[ok] - mean4, nrm))
            ray = c[ok] / np.maximum(r[sel][ok, None], 1e-9)
            cosi = np.abs(np.einsum("ij,ij->i", ray, nrm))
            # перепады больше 2 см у соседей - кромки, не шум
            span = np.maximum.reduce([np.abs(np.linalg.norm(q, axis=1) - r[sel])
                                      for q in (pu, pd, pl, pr)])[ok]
            flat = span < 0.05 + 0.02 * r[sel][ok]
            self.noise.append(np.c_[r[sel][ok][flat], np.degrees(np.arccos(np.clip(cosi[flat], 0, 1))),
                                    dev[flat]].astype(np.float32))
        # перепады вдоль столбца: доля точек между ближней и дальней поверхностью
        for a, b in ((up, dn), (lf, rt)):
            m = (a >= 0) & (b >= 0)
            ra, rb, rc = np.linalg.norm(xyz[a[m]], axis=1), np.linalg.norm(xyz[b[m]], axis=1), r[m]
            lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
            jump = hi - lo > 0.15
            between = jump & (rc > lo + 0.03) & (rc < hi - 0.03)
            self.jumps[0] += int(jump.sum())
            self.jumps[1] += int(between.sum())
            if between.any():
                t = (rc - lo)[between] / (hi - lo)[between]
                self.jump_pos += np.bincount(np.clip((t * 20).astype(int), 0, 19), minlength=20)

    def points(self):
        xyz = np.concatenate([k[0] for k in self.kept]) if self.kept else np.zeros((0, 3), np.float32)
        it = None
        if self.kept and self.kept[0][1] is not None:
            it = np.concatenate([k[1] for k in self.kept])
        return xyz, it

    def noise_table(self):
        if not self.noise:
            return None
        a = np.concatenate(self.noise)
        if len(a) > self.keep_pairs:
            a = a[self.rng.choice(len(a), self.keep_pairs, replace=False)]
        return a


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def noise_summary(tab: np.ndarray) -> dict:
    """СКО дальности по дальности и углу падения, мм. Отклонение от плоскости соседей
    пересчитывается в дальность: делим на cos(угла) и на sqrt(1,25) (шум четырёх соседей)."""
    out = {"range_bins_m": list(RANGE_BINS[:-1]), "incidence_bins_deg": list(INC_BINS[:-1]),
           "sigma_mm": [], "count": []}
    for r0, r1 in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        row, cnt = [], []
        for a0, a1 in zip(INC_BINS[:-1], INC_BINS[1:]):
            m = (tab[:, 0] >= r0) & (tab[:, 0] < r1) & (tab[:, 1] >= a0) & (tab[:, 1] < a1)
            if m.sum() < 200:
                row.append(None)
            else:
                cos = np.cos(np.radians(tab[m, 1]))
                row.append(round(_robust_sigma(tab[m, 2] / np.maximum(cos, 0.1)) / math.sqrt(1.25) * 1000, 3))
            cnt.append(int(m.sum()))
        out["sigma_mm"].append(row)
        out["count"].append(cnt)
    return out


def _memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:                                   # noqa: BLE001
        pass
    try:
        import resource
        import sys

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except Exception:                                   # noqa: BLE001
        return None


# ----------------------------------------------------------------------

def inspect_e57(path, out_dir, chunk: int = 2_000_000, keep_points: int = 1_500_000,
                keep_pairs: int = 300_000, sample: bool = True, sample_voxel_m: float = 0.04,
                sample_per_scan: int = 200_000, figures: bool = True, zip_result: bool = True,
                log=print) -> dict:
    import pye57

    path, out = Path(path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()
    report = {"file": {"name": path.name, "bytes": path.stat().st_size}, "scans": []}
    stations, samples = [], []
    with pye57.E57(str(path)) as e57:
        root = e57.root
        tree = node_tree(root)
        hdr = {k: tree.get(k) for k in ("formatName", "versionMajor", "versionMinor",
                                         "e57LibraryVersion", "coordinateMetadata")
               if isinstance(tree, dict) and k in tree}
        images = tree.get("images2D", []) if isinstance(tree, dict) else []
        report["file"].update(hdr)
        report["images2D"] = {"count": len(images) if isinstance(images, list) else None,
                              "bytes": _blob_bytes(images)}
        total_points = 0
        for i in range(e57.scan_count):
            h = e57.get_header(i)
            fields = [f for f in h.point_fields if f in SUPPORTED]
            cart = all(f in fields for f in ("cartesianX", "cartesianY", "cartesianZ"))
            if not cart:
                fields = [f for f in fields if not f.startswith("cartesian")]
            n_pts = h.point_count
            total_points += n_pts
            scan_tree = tree["data3D"][i] if isinstance(tree.get("data3D"), list) else {}
            rows = cols = None
            try:
                ib = h.indexBounds
                rows = int(ib["rowMaximum"].value()) - int(ib["rowMinimum"].value()) + 1
                cols = int(ib["columnMaximum"].value()) - int(ib["columnMinimum"].value()) + 1
            except Exception:                           # noqa: BLE001
                pass
            if "rowIndex" not in fields:
                rows = cols = None
            R = _quat_R(h.rotation) if h.has_pose() else np.eye(3)
            t = np.asarray(h.translation, float) if h.has_pose() else np.zeros(3)
            st = ScanStats(n_pts, rows, cols, keep_points, keep_pairs, rng)
            t0 = time.perf_counter()
            for d in iter_chunks(e57, i, fields, chunk):
                if cart:
                    xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
                    inv = d.get("cartesianInvalidState")
                else:
                    rr, az, el = d["sphericalRange"], d["sphericalAzimuth"], d["sphericalElevation"]
                    xyz = np.c_[rr * np.cos(el) * np.cos(az), rr * np.cos(el) * np.sin(az), rr * np.sin(el)]
                    inv = d.get("sphericalInvalidState")
                valid = np.ones(len(xyz), bool) if inv is None else (inv == 0)
                valid &= np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
                st.add(xyz, d.get("intensity"), d.get("rowIndex"), d.get("columnIndex"), valid)
            secs = time.perf_counter() - t0
            xyz_l, inten = st.points()
            r_l = np.linalg.norm(xyz_l, axis=1)
            world_coords = bool(len(r_l) and not h.has_pose() and np.median(r_l) > 50)
            xyz_w = (xyz_l.astype(np.float64) @ R.T + t)
            stations.append({"center": t, "points": xyz_w})
            tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
            info = {
                "index": i, "name": scan_tree.get("name") if isinstance(scan_tree, dict) else None,
                "points": int(n_pts), "valid": int(st.valid),
                "valid_fraction": round(st.valid / max(st.n, 1), 4),
                "fields": field_types(h), "has_color": "colorRed" in h.point_fields,
                "grid_rows_cols": [rows, cols] if rows else None,
                "grid_fill": round(n_pts / (rows * cols), 4) if rows else None,
                "azimuth_step_deg": round(float(np.median(st.az_step)), 5) if st.az_step else None,
                "elevation_step_deg": round(float(np.median(st.el_step)), 5) if st.el_step else None,
                "pose": {"translation": t.round(4).tolist(),
                         "rotation_wxyz": [round(float(v), 6) for v in h.rotation]} if h.has_pose() else None,
                "tilt_deg": round(tilt, 3),
                "points_in_world_coords": world_coords,
                "range_m": {"min": round(float(r_l.min()), 3) if len(r_l) else None,
                            "p50": round(float(np.median(r_l)), 3) if len(r_l) else None,
                            "p99": round(float(np.percentile(r_l, 99)), 3) if len(r_l) else None,
                            "max": round(float(r_l.max()), 3) if len(r_l) else None},
                "intensity": None if inten is None or not len(inten) else {
                    "min": float(inten.min()), "p1": float(np.percentile(inten, 1)),
                    "p50": float(np.median(inten)), "p99": float(np.percentile(inten, 99)),
                    "max": float(inten.max())},
                "jumps": {"count": st.jumps[0], "between_fraction":
                          round(st.jumps[1] / max(st.jumps[0], 1), 4),
                          "between_position_hist": st.jump_pos.tolist()},
                "read_seconds": round(secs, 2), "read_points_per_s": int(n_pts / max(secs, 1e-9)),
            }
            from .rasterize import _plane_z

            rel = xyz_w[:, 2] - t[2]
            fz, cz = _plane_z(rel, -2.5, -0.3), _plane_z(rel, 0.3, 4.0)
            info["floor_rel_m"] = None if fz is None else round(fz, 3)
            info["ceiling_rel_m"] = None if cz is None else round(cz, 3)
            if fz is not None:
                fl = np.abs(rel - fz) < 0.05
                info["floor_flatness_mm"] = round(float(np.std(rel[fl] - fz)) * 1000, 2) if fl.any() else None
            tab = st.noise_table()
            if tab is not None:
                info["noise"] = noise_summary(tab)
                st.noise_tab = tab
            report["scans"].append(info)
            st.kept = []                                # точки уже в stations
            samples.append((st, inten))
            log(f"станция {i + 1}/{e57.scan_count}: {n_pts / 1e6:.1f} млн точек, {secs:.0f} с")
    total_s = time.perf_counter() - t_all
    img = report["images2D"]["bytes"] or 0
    report["file"]["points_total"] = int(total_points)
    report["file"]["bytes_per_point"] = round((report["file"]["bytes"] - img) / max(total_points, 1), 2)
    report["performance"] = {"seconds_total": round(total_s, 1),
                             "points_per_s": int(total_points / max(total_s, 1e-9)),
                             "peak_memory_mb": _memory_mb(), "chunk_points": chunk}
    floors = [c["center"][2] + s["floor_rel_m"] for c, s in zip(stations, report["scans"])
              if s["floor_rel_m"] is not None]
    report["levelling"] = {"floor_z_std_mm": round(float(np.std(floors)) * 1000, 2) if floors else None,
                           "station_height_m": [s["floor_rel_m"] and round(-s["floor_rel_m"], 3)
                                                for s in report["scans"]]}
    plan = _plan(stations, out) if stations else None
    if plan:
        report["plan"] = plan["info"]
    if figures:
        _figures(report, stations, samples, plan, out)
    if sample:
        _write_sample(stations, samples, out / "sample.npz", sample_voxel_m, sample_per_scan, rng)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_js),
                                     encoding="utf-8")
    (out / "header_tree.json").write_text(json.dumps(tree, indent=1, ensure_ascii=False, default=_js),
                                          encoding="utf-8")
    (out / "summary.txt").write_text(_summary(report), encoding="utf-8")
    if zip_result:
        report["zip"] = shutil.make_archive(str(out), "zip", root_dir=out)
    return report


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _plan(stations, out: Path) -> dict | None:
    """Псевдочертёж в СК файла: кадр по 1-99 % точек, пиксель 1 см (не больше 2048 px)."""
    import cv2

    from .groundtruth import RasterFrame
    from .rasterize import free_space_input

    allp = np.concatenate([s["points"][:: max(1, len(s["points"]) // 200_000)] for s in stations])
    if len(allp) < 1000:
        return None
    lo = np.percentile(allp[:, :2], 1, axis=0) - 1.0
    hi = np.percentile(allp[:, :2], 99, axis=0) + 1.0
    px = max(0.01, float((hi - lo).max()) / 2048)
    w, h = int(math.ceil((hi[0] - lo[0]) / px)), int(math.ceil((hi[1] - lo[1]) / px))
    frame = RasterFrame(float(lo[0]), float(lo[1]), px, w, h)
    res = free_space_input(stations, frame)
    cv2.imwrite(str(out / "input.png"), res["image"])
    free_m2 = float(res["free"].sum()) * px * px
    return {"frame": frame, "res": res,
            "info": {"pixel_mm": round(px * 1000, 2), "size_px": [h, w],
                     "free_area_m2": round(free_m2, 2), "stations": res["stations"]}}


def _write_sample(stations, samples, path: Path, voxel: float, per_scan: int, rng) -> None:
    xs, its, sid = [], [], []
    for k, (s, (_, inten)) in enumerate(zip(stations, samples)):
        p = s["points"]
        key = np.floor(p / voxel).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        if len(first) > per_scan:
            first = rng.choice(first, per_scan, replace=False)
        xs.append(p[first].astype(np.float32))
        its.append(np.zeros(len(first), np.float32) if inten is None else inten[first])
        sid.append(np.full(len(first), k, np.uint8))
    np.savez_compressed(path, xyz=np.concatenate(xs), intensity=np.concatenate(its).astype(np.float16),
                        station=np.concatenate(sid),
                        station_xyz=np.array([s["center"] for s in stations], np.float64))


def _summary(r: dict) -> str:
    f = r["file"]
    lines = [f"{f['name']}: {f['bytes'] / 1e9:.2f} ГБ, {f['points_total'] / 1e6:.1f} млн точек, "
             f"{f['bytes_per_point']} Б/точку, станций {len(r['scans'])}, "
             f"снимков {r['images2D']['count']} ({r['images2D']['bytes'] / 1e6:.0f} МБ)",
             f"чтение {r['performance']['seconds_total']} с ({r['performance']['points_per_s'] / 1e6:.2f} "
             f"млн точек/с), память {r['performance']['peak_memory_mb']} МБ"]
    for s in r["scans"]:
        lines.append(
            f"  {s['index'] + 1:>2} {s['name'] or ''}: {s['points'] / 1e6:.1f} млн, сетка {s['grid_rows_cols']}, "
            f"шаг {s['azimuth_step_deg']}°/{s['elevation_step_deg']}°, без отклика "
            f"{(1 - s['valid_fraction']) * 100:.1f} %, наклон {s['tilt_deg']}°, "
            f"пол {s['floor_rel_m']} м, потолок {s['ceiling_rel_m']} м, "
            f"перепады «между» {s['jumps']['between_fraction'] * 100:.1f} %")
    lines.append(f"поля: {r['scans'][0]['fields'] if r['scans'] else '-'}")
    return "\n".join(lines) + "\n"


def _figures(report, stations, samples, plan, out: Path) -> None:
    from .debug import AQUA, BLUE, INK, INK2, ORANGE, RED, SURFACE, YELLOW, _plt

    plt = _plt()
    # план: срез и вход
    if plan:
        from .rasterize import _slice_density

        fr, res = plan["frame"], plan["res"]
        dens = _slice_density(stations, fr)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
        axes[0].imshow(1 - dens * 0.85, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        axes[1].imshow(res["image"], cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
        for ax, title in zip(axes, ("Срез 1,0–1,6 м", "Вход")):
            for k, s in enumerate(stations):
                i, j = fr.xy_to_ij(s["center"][0], s["center"][1])
                ax.plot(j, i, "o", ms=5, mfc="white", mec=INK, mew=1)
                ax.annotate(str(k + 1), (j, i), xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color=INK2)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "plan.png", dpi=90)
        plt.close(fig)
    # панорамы: строка на станцию, дальность | интенсивность | без отклика
    pans = [(k, st) for k, (st, _) in enumerate(samples) if st.pano is not None]
    for page in range(0, len(pans), 8):
        part = pans[page:page + 8]
        fig, axes = plt.subplots(len(part), 3, figsize=(16, 1.9 * len(part) + 0.4), squeeze=False)
        for row, (k, st) in enumerate(part):
            P = st.pano
            n = np.maximum(P["n"], 1)
            rng_img = np.where(P["n"] > 0, P["r"] / n, np.nan)
            int_img = np.where(P["n"] > 0, P["i"] / n, np.nan)
            miss = np.where(P["all"] > 0, 1 - P["n"] / np.maximum(P["all"], 1), np.nan)
            for c, (img, cmap, title) in enumerate(((rng_img, "Blues_r", "Дальность"),
                                                    (int_img, "gray", "Интенсивность"),
                                                    (miss, "Reds", "Без отклика"))):
                ax = axes[row, c]
                vmax = np.nanpercentile(img, 99) if np.isfinite(img).any() else 1
                ax.imshow(img, cmap=cmap, aspect="auto", interpolation="antialiased", vmin=0,
                          vmax=vmax if c < 2 else 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(title)
            axes[row, 0].set_ylabel(str(k + 1), rotation=0, ha="right", va="center", color=INK2)
        fig.tight_layout()
        fig.savefig(out / f"panoramas_{page // 8 + 1}.png", dpi=80)
        plt.close(fig)
    # сенсор: дальности, интенсивность от дальности, шум от угла, перепады
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    colors = (BLUE, ORANGE, AQUA, YELLOW, RED, INK)
    rmax = max(float(np.percentile(np.linalg.norm(s["points"] - s["center"], axis=1), 99.5))
               for s in stations[:6] if len(s["points"]))
    step = max(0.05, rmax / 60)
    for k, s in enumerate(stations[:6]):
        r = np.linalg.norm(s["points"] - s["center"], axis=1)
        axes[0].hist(r, bins=np.arange(0, rmax + step, step), histtype="step", color=colors[k % 6],
                     label=str(k + 1))
        it = samples[k][1]
        if it is not None and len(it):
            b = np.arange(0, rmax + 2 * step, 2 * step)
            idx = np.digitize(r, b)
            med = [np.median(it[idx == j]) if (idx == j).sum() > 200 else np.nan for j in range(1, len(b))]
            axes[1].plot(b[:-1] + step, med, color=colors[k % 6], lw=1.5)
    axes[0].set_title("Дальность")
    axes[0].set_xlabel("м")
    axes[0].legend(title="станция", fontsize=8)
    axes[1].set_title("Интенсивность (медиана)")
    axes[1].set_xlabel("дальность, м")
    tabs = [st.noise_tab for st, _ in samples if hasattr(st, "noise_tab")]
    if tabs:
        tab = np.concatenate(tabs)
        for (r0, r1), col in zip(((0, 3), (3, 6), (6, 12), (12, 40)), (BLUE, ORANGE, AQUA, RED)):
            xs, ys = [], []
            for a0 in range(0, 85, 5):
                m = (tab[:, 0] >= r0) & (tab[:, 0] < r1) & (tab[:, 1] >= a0) & (tab[:, 1] < a0 + 5)
                if m.sum() > 300:
                    cos = np.cos(np.radians(tab[m, 1]))
                    xs.append(a0 + 2.5)
                    ys.append(_robust_sigma(tab[m, 2] / np.maximum(cos, 0.1)) / math.sqrt(1.25) * 1000)
            if xs:
                axes[2].plot(xs, ys, "o-", color=col, ms=3, lw=1.5, label=f"{r0}–{r1} м")
        axes[2].set_ylim(bottom=0)
        axes[2].legend(fontsize=8)
    axes[2].set_title("Шум дальности, мм")
    axes[2].set_xlabel("угол падения, °")
    pos = np.sum([s["jumps"]["between_position_hist"] for s in report["scans"]], axis=0)
    axes[3].bar(np.arange(20) / 20 + 0.025, pos, width=0.045, color=BLUE)
    axes[3].set_title("Точки на перепадах")
    axes[3].set_xlabel("0 - ближняя поверхность, 1 - дальняя")
    fig.tight_layout()
    fig.savefig(out / "sensor.png", dpi=90)
    plt.close(fig)
    del SURFACE
