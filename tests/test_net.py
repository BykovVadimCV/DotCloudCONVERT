"""Сеть плана: формы голов, конечные потери, склейка окон при применении (нужен torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simscan.infer import predict  # noqa: E402
from simscan.net import PlanNet, confusion, iou_from_confusion, losses  # noqa: E402


def _batch(h=64, w=80):
    sem = torch.zeros(1, h, w, dtype=torch.long)
    sem[:, :, 30:34] = 2
    sem[:, 20:24, 30:34] = 3
    sem[:, :2] = 255
    orient = torch.zeros(1, 2, h, w)
    orient[:, 0] = 1.0
    return {"input": torch.rand(1, 8, h, w), "sem": sem, "dist": torch.rand(1, 1, h, w), "orient": orient,
            "heights": torch.rand(1, 2, h, w), "mask_orient": sem == 2, "mask_heights": sem == 3}


def test_forward_and_losses():
    m = PlanNet(width=16)
    b = _batch()
    o = m(b["input"])                                    # размер не кратен 32 - дополняется внутри
    assert o["sem"].shape == (1, 8, 64, 80) and o["orient"].shape == (1, 2, 64, 80)
    parts = losses(o, b)
    assert all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()
    cm = confusion(o["sem"].argmax(1), b["sem"])
    assert int(cm.sum()) == int((b["sem"] != 255).sum()) and len(iou_from_confusion(cm)) == 8


def test_predict_tiles_cover_raster():
    m = PlanNet(width=16).eval()
    x = np.random.rand(8, 150, 210).astype(np.float32)
    r = predict(m, torch.device("cpu"), x, tile=96, overlap=32)
    assert r["prob"].shape == (8, 150, 210) and np.allclose(r["prob"].sum(0), 1, atol=1e-3)
    assert np.isfinite(r["dist"]).all() and np.allclose(np.linalg.norm(r["orient"], axis=0), 1, atol=1e-3)
