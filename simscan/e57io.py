"""Запись синтетических сканов в E57 через pye57 - тем же путём, каким конвертер читает реальные файлы.

Особенности pye57 0.4.x (сверено с исходником e57.py):
  - write_scan_raw пишет только декартовы координаты, точность E57_SINGLE (float32).
    Поэтому точки пишутся в локальной СК станции (до ~100 м шаг float32 < 0,01 мм),
    а поза (кватернион w,x,y,z + перенос в float64) - в заголовке скана;
  - read_scan выбрасывает точки с ненулевым InvalidState и применяет позу;
  - cartesianBounds, которые пишет pye57, посчитаны поворотом двух углов габарита -
    при повороте скана они неверны; конвертеру на них опираться нельзя.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scanner import ScanResult
from .transform import quat_from_matrix

INVALID_DIRECTION_ONLY = 1   # E57: координаты задают только направление (дальности нет)


def scan_payload(scan: ScanResult, keep_invalid: bool) -> tuple[dict, np.ndarray]:
    """Поля E57 для одного скана и маска записанных точек (в порядке сетки)."""
    written = np.ones(len(scan.valid), bool) if keep_invalid else scan.valid.copy()
    xyz = scan.xyz_local[written]
    valid = scan.valid[written]
    if not valid.any():
        raise ValueError(f"станция {scan.station_id}: ни одной действительной точки")
    lo, hi = xyz[valid].min(0), xyz[valid].max(0)
    # направления без дальности держим внутри габарита, иначе libE57 отвергнет значение
    xyz = np.where(valid[:, None], xyz, np.clip(xyz, lo, hi))
    data = {
        "cartesianX": xyz[:, 0], "cartesianY": xyz[:, 1], "cartesianZ": xyz[:, 2],
        "intensity": scan.intensity[written],
        "rowIndex": scan.row[written], "columnIndex": scan.col[written],
    }
    if keep_invalid:
        data["cartesianInvalidState"] = np.where(valid, 0, INVALID_DIRECTION_ONLY).astype(np.int8)
    return data, written


def write_e57(path: str | Path, scans: list[ScanResult], poses: list[tuple[np.ndarray, np.ndarray]],
              names: list[str], keep_invalid: bool = True) -> list[np.ndarray]:
    """Пишет сканы со своими позами. Возвращает маски записанных точек по каждому скану."""
    import pye57

    path = Path(path)
    if path.exists():
        path.unlink()
    masks = []
    with pye57.E57(str(path), mode="w") as e57:
        for scan, (R, t), name in zip(scans, poses, names):
            data, written = scan_payload(scan, keep_invalid)
            e57.write_scan_raw(data, name=name, rotation=quat_from_matrix(R),
                               translation=np.asarray(t, float))
            masks.append(written)
    return masks


def write_merged_e57(path: str | Path, points_world: np.ndarray, intensity: np.ndarray) -> None:
    """Слитое облако одним сканом (как «экспорт без станций»).

    Точки пишутся относительно округлённого центра, центр - в перенос позы:
    глобальные координаты в float32 потеряли бы миллиметры уже на сотнях метров.
    """
    import pye57

    path = Path(path)
    if path.exists():
        path.unlink()
    shift = np.round(points_world.mean(0), 0)
    local = points_world - shift
    with pye57.E57(str(path), mode="w") as e57:
        e57.write_scan_raw({"cartesianX": local[:, 0], "cartesianY": local[:, 1],
                            "cartesianZ": local[:, 2], "intensity": intensity},
                           name="merged", rotation=np.array([1.0, 0, 0, 0]), translation=shift)


def read_e57_world(path: str | Path) -> list[dict]:
    """Читает все сканы в глобальной СК (как это будет делать конвертер)."""
    import pye57

    out = []
    with pye57.E57(str(path)) as e57:
        for i in range(e57.scan_count):
            d = e57.read_scan(i, intensity=True, row_column=True, transform=True,
                              ignore_missing_fields=True)
            xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
            out.append({"xyz": xyz, "intensity": d.get("intensity"),
                        "row": d.get("rowIndex"), "col": d.get("columnIndex"),
                        "position": e57.scan_position(i)[0],
                        "name": e57.get_header(i).name if hasattr(e57.get_header(i), "name") else str(i)})
    return out
