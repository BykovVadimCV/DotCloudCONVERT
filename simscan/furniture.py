"""Процедурная мебель: предметы из частей, чтобы лучи проходили там же, где в жизни.

Система координат предмета (как в interior._local_box): du - вдоль стены, dv - от стены
в комнату, z - от пола. base - габарит предмета (Box, стоящий спиной к стене).
Функции возвращают (boxes, prims): параллелепипеды и не-параллелепипеды (цилиндры).
"""
from __future__ import annotations

import math

import numpy as np

from .layout import Box


def lb(base: Box, du: float, dv: float, z0: float, size) -> Box:
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    x = base.center[0] + du * c - dv * s
    y = base.center[1] + du * s + dv * c
    return Box((x, y, z0 + size[2] / 2), tuple(float(v) for v in size), base.yaw)


def _legs(base, w, d, h, t=0.04, inset=0.04):
    return [lb(base, su * (w / 2 - inset - t / 2), sv * (d / 2 - inset - t / 2), 0.0, (t, t, h))
            for su in (-1, 1) for sv in (-1, 1)]


def chair(base: Box, rng) -> list:
    """Стул: ножки, сиденье, спинка - под сиденьем и между ножками лучи проходят."""
    w, d = base.size[0], base.size[1]
    seat_z = float(rng.uniform(0.42, 0.47))
    boxes = _legs(base, w, d, seat_z - 0.04, t=0.035, inset=0.02)
    boxes.append(lb(base, 0, 0, seat_z - 0.04, (w, d, 0.04)))
    back_h = float(rng.uniform(0.35, 0.5))
    boxes.append(lb(base, 0, -d / 2 + 0.015, seat_z, (w, 0.03, back_h)))
    return boxes


def sofa(base: Box, rng) -> list:
    w, d, h = base.size
    gap = float(rng.uniform(0.04, 0.15))
    boxes = _legs(base, w, d, gap, t=0.05, inset=0.05)
    boxes.append(lb(base, 0, 0.1, gap, (w - 0.36, d - 0.2, 0.42 - gap)))       # сиденье
    boxes.append(lb(base, 0, -d / 2 + 0.1, gap, (w, 0.2, h - gap)))             # спинка
    for su in (-1, 1):                                                          # подлокотники
        boxes.append(lb(base, su * (w / 2 - 0.09), 0.1, gap, (0.18, d - 0.2, 0.62 - gap)))
    return boxes


def bed(base: Box, rng) -> list:
    w, d, h = base.size
    on_legs = rng.random() < 0.6
    gap = float(rng.uniform(0.18, 0.3)) if on_legs else 0.0
    boxes = _legs(base, w, d, gap, t=0.06) if on_legs else []
    boxes.append(lb(base, 0, 0, gap, (w, d, 0.15)))                             # рама
    boxes.append(lb(base, 0, 0.01, gap + 0.15, (w - 0.04, d - 0.06, max(0.12, h - gap - 0.15))))
    boxes.append(lb(base, 0, -d / 2 + 0.03, 0.0, (w, 0.06, float(rng.uniform(0.9, 1.2)))))
    return boxes


def shelf(base: Box, rng) -> list:
    """Открытый стеллаж: боковины, полки, книги; задняя стенка не всегда."""
    w, d, h = base.size
    t = 0.02
    boxes = [lb(base, su * (w / 2 - t / 2), 0, 0.0, (t, d, h)) for su in (-1, 1)]
    boxes.append(lb(base, 0, 0, h - t, (w - 2 * t, d, t)))
    boxes.append(lb(base, 0, 0.02, 0.0, (w - 2 * t, d - 0.04, 0.06)))          # цоколь
    if rng.random() < 0.5:
        boxes.append(lb(base, 0, -d / 2 + 0.003, 0.06, (w - 2 * t, 0.006, h - 0.06 - t)))
    step = float(rng.uniform(0.3, 0.4))
    z = 0.06
    while z + step < h - t:
        boxes.append(lb(base, 0, 0, z + step - t, (w - 2 * t, d, t)))
        # книги блоками: заполнение 30-90 % полки
        u = -w / 2 + t
        while u < w / 2 - t - 0.1:
            bw = float(rng.uniform(0.08, 0.35))
            if rng.random() < 0.65 and u + bw < w / 2 - t:
                bh = float(min(step - 0.04, rng.uniform(0.15, 0.3)))
                bd = float(min(d - 0.02, rng.uniform(0.15, 0.25)))
                boxes.append(lb(base, u + bw / 2, -d / 2 + bd / 2 + 0.01, z, (bw, bd, bh)))
            u += bw + float(rng.uniform(0.0, 0.1))
        z += step
    return boxes


def wardrobe(base: Box, rng) -> list:
    w, d, h = base.size
    return [lb(base, 0, -0.02, 0.0, (w - 0.02, d - 0.06, 0.08)),               # утопленный цоколь
            lb(base, 0, 0, 0.08, (w, d, h - 0.08))]


def tv_stand(base: Box, rng) -> list:
    w, d, h = base.size
    gap = 0.1
    boxes = _legs(base, w, d, gap, t=0.04)
    boxes.append(lb(base, 0, 0, gap, (w, d, h - gap)))
    if rng.random() < 0.6:
        tw = w * float(rng.uniform(0.5, 0.8))
        boxes.append(lb(base, 0, -d / 2 + 0.08, h, (tw, 0.05, tw * 0.58)))
    return boxes


def counter(base: Box, rng) -> list:
    """Кухонная тумба: утопленный цоколь, корпус, столешница со свесом."""
    w, d, h = base.size
    return [lb(base, 0, -0.04, 0.0, (w, d - 0.08, 0.1)),
            lb(base, 0, -0.01, 0.1, (w, d - 0.02, h - 0.14)),
            lb(base, 0, 0.01, h - 0.04, (w, d + 0.02, 0.04))]


def shoe_rack(base: Box, rng) -> list:
    w, d, h = base.size
    t = 0.02
    boxes = [lb(base, su * (w / 2 - t / 2), 0, 0.0, (t, d, h)) for su in (-1, 1)]
    n = max(2, int(h / 0.25))
    for k in range(n + 1):
        boxes.append(lb(base, 0, 0, min(h - t, k * h / n), (w - 2 * t, d, t)))
    return boxes


def table(base: Box, rng) -> list:
    w, d, h = base.size
    return [lb(base, 0, 0, h - 0.03, (w, d, 0.03)), *_legs(base, w, d, h - 0.03, t=0.05, inset=0.03)]


def plant(x: float, y: float, rng):
    """Растение: горшок, стебель и облако листьев - насквозь просвечивает."""
    r = float(rng.uniform(0.12, 0.2))
    ph = float(rng.uniform(0.25, 0.45))
    stem = float(rng.uniform(0.3, 0.9))
    prims = [{"type": "cylinder", "center": [x, y, ph / 2], "radius": r, "height": ph},
             {"type": "cylinder", "center": [x, y, ph + stem / 2], "radius": 0.015, "height": stem}]
    boxes = []
    crown_r = float(rng.uniform(0.2, 0.45))
    cz = ph + stem
    for _ in range(int(rng.integers(30, 70))):
        v = rng.normal(0, 1, 3)
        v /= np.linalg.norm(v)
        p = np.array([x, y, cz]) + v * crown_r * rng.uniform(0.2, 1.0) * np.array([1, 1, 0.8])
        s = rng.uniform(0.04, 0.12)
        boxes.append(Box(tuple(p), (float(s), float(s * 0.5), 0.01), float(rng.uniform(0, math.pi))))
    return boxes, prims


def coat_rack(x: float, y: float, rng):
    prims = [{"type": "cylinder", "center": [x, y, 0.015], "radius": 0.2, "height": 0.03},
             {"type": "cylinder", "center": [x, y, 0.9], "radius": 0.02, "height": 1.8}]
    boxes = []
    for k in range(int(rng.integers(2, 6))):
        a = rng.uniform(0, 2 * math.pi)
        h = float(rng.uniform(0.6, 0.95))
        boxes.append(Box((x + 0.15 * math.cos(a), y + 0.15 * math.sin(a), 1.75 - h / 2),
                         (0.4, 0.12, h), float(a + math.pi / 2)))
    return boxes, prims


def radiator(wall, s: float, face: float, side: int, width: float, height: float, rng) -> list:
    """Радиатор: секционный (рёбра с зазорами) или панельный."""
    yaw = wall.yaw
    z0 = 0.1
    if rng.random() < 0.5:
        center = wall.point(s, face + side * 0.08)
        return [Box((*center, z0 + height / 2), (width, 0.08, height), yaw)]
    boxes = []
    n = max(3, int(width / 0.08))
    pitch = width / n
    for k in range(n):
        sk = s - width / 2 + (k + 0.5) * pitch
        c = wall.point(sk, face + side * 0.08)
        boxes.append(Box((*c, z0 + height / 2), (pitch - 0.012, 0.08, height), yaw))
    return boxes
