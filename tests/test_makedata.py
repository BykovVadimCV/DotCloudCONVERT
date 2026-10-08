"""make-dataset: сцена -> датасет без скана, продолжение с места обрыва, режим «под отделку», zip."""
import json
import zipfile

from simscan.config import load_config
from simscan.makedata import make_dataset, pack, rebuild_splits


def test_make_dataset_resume_and_pack(tmp_path):
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5}, "interior": {"p_bare": 1.0}})
    root = tmp_path / "ds"
    r = make_dataset(cfg, root, count=1, seed=5, log=lambda *a: None)
    assert r["ok"] == 1 and r["errors"] == 0 and sum(r["splits"].values()) == 1
    d = root / "scenes" / "scene_00000" / "L1"
    for f in ("input.npy", "sem.png", "dist.npy", "orient.npy", "heights.npy", "scene.json"):
        assert (d / f).exists(), f
    assert not (root / "_work").exists()                       # скан удалён
    sc = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert sc["bare"] is True and "thin" in sc["aug"]
    r2 = make_dataset(cfg, root, count=1, seed=5, log=lambda *a: None)
    assert r2["ok"] == 0                                        # готовая сцена пропущена
    assert rebuild_splits(root) == r["splits"]
    names = zipfile.ZipFile(pack(root, tmp_path / "ds.zip")).namelist()
    assert "ds/meta.json" in names and "ds/scenes/scene_00000/L1/input.npy" in names
    assert not any(n.endswith("preview.png") for n in names)
