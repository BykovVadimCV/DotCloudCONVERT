"""Собственный читатель E57 против libE57 (pye57): те же значения во всех типах полей."""
import numpy as np
import pye57

from simscan.e57read import E57Reader

from conftest import box_room_config
from e57_scaled_writer import write_scaled


def _read_all(path, index, chunk):
    r = E57Reader(path)
    s = r.scans[index]
    parts = {k: [] for k in s.field_names}
    for c in r.iter_points(index, s.field_names, chunk):
        for k, v in c.items():
            parts[k].append(v)
    return r, {k: np.concatenate(v) for k, v in parts.items()}


def test_matches_pye57_on_float_file(tmp_path):
    from simscan.generate import generate_scene

    cfg = box_room_config(5.0, 3.0, layout={"source": "simscan"}, export={"free_space_input": False})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    path = tmp_path / "s" / "scan.e57"
    with pye57.E57(str(path)) as e:
        ref = e.read_scan_raw(0)
        rot, tr = e.get_header(0).rotation, e.get_header(0).translation
    r, mine = _read_all(path, 0, 7_777)                 # кусок не кратен пакету
    for k, v in ref.items():
        assert np.array_equal(mine[k].astype(np.float64), v.astype(np.float64)), k
    assert np.allclose(r.scans[0].rotation, rot) and np.allclose(r.scans[0].translation, tr)


def test_scaled_integer_color_and_odd_bit_widths(tmp_path):
    path = str(tmp_path / "scaled.e57")
    ref = write_scaled(path, n=120_000)
    r, mine = _read_all(path, 0, 50_000)
    widths = {f.name: f.bits for f in r.scans[0].fields}
    assert widths["cartesianX"] == 20 and widths["intensity"] == 11 and widths["returnIndex"] == 0
    assert widths["cartesianInvalidState"] == 2
    for k, v in ref.items():
        assert np.allclose(mine[k], v, atol=1e-6), k
    assert r.scans[0].tree["sensorSerialNumber"] == "<скрыто>"
