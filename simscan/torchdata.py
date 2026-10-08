"""Загрузчик датасета плана (раскладка simscan/netinput.py) для обучения.

    from simscan.torchdata import FloorplanSamples, torch_dataset
    ds = torch_dataset("dataset", "train", crop=512)      # нужен torch
    s = FloorplanSamples("dataset", "train")[0]           # то же на numpy, без torch

Образец: input [8, h, w] float32; sem [h, w] int64 (255 - ignore); dist [1, h, w];
orient [2, h, w]; heights [2, h, w]; mask_orient, mask_heights [h, w] bool.

Аугментации на растре - только точные: случайный кроп, отражения и повороты на 90°
(без интерполяции), выпадение каналов. Повороты на произвольный угол, прореживание,
сдвиг пола - на облаке до растеризации (netinput.rasterize_e57: thin, z_shift), иначе
появятся артефакты, которых нет в реальных данных. Ориентация (cos 2θ, sin 2θ) пересчитывается
вместе с отражением и поворотом.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IGNORE = 255
WALLISH = (2, 3, 4, 5)
OPENINGS = (3, 4, 5)


def load_level(d: Path) -> dict:
    import cv2

    d = Path(d)
    out = {"input": np.load(d / "input.npy").astype(np.float32)}
    if (d / "sem.png").exists():
        out["sem"] = cv2.imread(str(d / "sem.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)
        out["dist"] = np.load(d / "dist.npy").astype(np.float32)
        out["orient"] = np.load(d / "orient.npy").astype(np.float32)
        out["heights"] = np.load(d / "heights.npy").astype(np.float32)
    return out


def _flip_x(s: dict) -> dict:                      # столбцы: x -> -x, угол θ -> π - θ
    out = {k: (v[..., ::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"][1] *= -1
    return out


def _flip_y(s: dict) -> dict:                      # строки: y -> -y, θ -> -θ
    out = {k: (v[..., ::-1, :].copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"][1] *= -1
    return out


def _rot90(s: dict) -> dict:                       # против часовой на 90°: θ -> θ + 90°, 2θ -> 2θ + 180°
    out = {k: (np.rot90(v, 1, axes=(-2, -1)).copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"] *= -1
    return out


class FloorplanSamples:
    """Образцы на numpy. crop - сторона кропа (None - уровень целиком); train - аугментации."""

    def __init__(self, root, split: str = "train", crop: int | None = 512, train: bool = True,
                 channel_dropout: float = 0.1, seed: int = 0):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        f = self.root / "splits" / f"{split}.txt"
        self.items = [x for x in f.read_text(encoding="utf-8").split() if x] if f.exists() else []
        self.crop, self.train, self.p_drop = crop, train, channel_dropout
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, s: dict) -> dict:
        C = self.crop
        H, W = s["input"].shape[-2:]
        ph, pw = max(0, C - H), max(0, C - W)
        if ph or pw:                                   # мал - дополнить: вход 0, разметка ignore
            pad = lambda a, v: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)], constant_values=v)  # noqa: E731
            s = {k: pad(v, IGNORE if k == "sem" else 0) for k, v in s.items()}
            H, W = H + ph, W + pw
        if self.train:
            # кроп чаще с размеченным (не только улицей)
            for _ in range(10):
                y = int(self.rng.integers(0, H - C + 1))
                x = int(self.rng.integers(0, W - C + 1))
                if "sem" not in s or ((s["sem"][y:y + C, x:x + C] % 255) > 0).mean() > 0.05:
                    break
        else:
            y, x = (H - C) // 2, (W - C) // 2
        return {k: v[..., y:y + C, x:x + C] for k, v in s.items()}

    def __getitem__(self, idx: int) -> dict:
        s = load_level(self.root / "scenes" / self.items[idx])
        if self.crop:
            s = self._crop(s)
        if self.train:
            if self.rng.random() < 0.5:
                s = _flip_x(s)
            if self.rng.random() < 0.5:
                s = _flip_y(s)
            for _ in range(int(self.rng.integers(0, 4))):
                s = _rot90(s)
            drop = self.rng.random(s["input"].shape[0]) < self.p_drop
            s["input"][drop] = 0.0
        if "sem" in s:
            sem = s["sem"]
            s["mask_orient"] = np.isin(sem, WALLISH)
            s["mask_heights"] = np.isin(sem, OPENINGS)
        s["id"] = self.items[idx]
        return s


def torch_dataset(root, split: str = "train", **kw):
    """Обёртка torch.utils.data.Dataset над FloorplanSamples (нужен torch)."""
    import torch
    from torch.utils.data import Dataset

    base = FloorplanSamples(root, split, **kw)

    class _DS(Dataset):
        def __len__(self):
            return len(base)

        def __getitem__(self, i):
            s = base[i]
            return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                    for k, v in s.items()}

    return _DS()
