"""Наполнение помещений: мебель, инженерные элементы, колонны, зеркала, окружение.

Всё задаётся ориентированными параллелепипедами (Box). Для эталонного плана
важны только стены и колонны; остальное - помехи, которые конвертер должен
пережить (мебель у стен, шкафы до потолка, радиаторы под окнами, короба).
"""
from __future__ import annotations

import math

import numpy as np

from .config import InteriorConfig
from .layout import Box, Item, Layout, Room, Wall

RES = 0.05  # шаг сетки занятости помещения, м


class RoomGrid:
    """Сетка занятости одного помещения по его чистому прямоугольнику."""

    def __init__(self, room: Room):
        self.x0, self.y0, self.x1, self.y1 = room.clear
        self.nx = max(1, int(round((self.x1 - self.x0) / RES)))
        self.ny = max(1, int(round((self.y1 - self.y0) / RES)))
        self.occ = np.zeros((self.ny, self.nx), bool)        # ничего ставить нельзя
        self.no_tall = np.zeros((self.ny, self.nx), bool)    # нельзя высокое (окна, зеркала)

    def _slices(self, rect):
        xa, ya, xb, yb = rect
        i0 = int(math.floor((xa - self.x0) / RES + 1e-6))
        i1 = int(math.ceil((xb - self.x0) / RES - 1e-6))
        j0 = int(math.floor((ya - self.y0) / RES + 1e-6))
        j1 = int(math.ceil((yb - self.y0) / RES - 1e-6))
        return (slice(max(j0, 0), max(min(j1, self.ny), 0)),
                slice(max(i0, 0), max(min(i1, self.nx), 0)))

    def inside(self, rect) -> bool:
        xa, ya, xb, yb = rect
        return (xa >= self.x0 - 1e-6 and ya >= self.y0 - 1e-6
                and xb <= self.x1 + 1e-6 and yb <= self.y1 + 1e-6)

    def is_free(self, rect, tall: bool) -> bool:
        if not self.inside(rect):
            return False
        sl = self._slices(rect)
        if self.occ[sl].any():
            return False
        return not (tall and self.no_tall[sl].any())

    def mark(self, rect, margin: float = 0.05, grid: str = "occ") -> None:
        xa, ya, xb, yb = rect
        getattr(self, grid)[self._slices((xa - margin, ya - margin, xb + margin, yb + margin))] = True

    def free_points(self, clearance: float) -> np.ndarray:
        """Центры свободных клеток не ближе clearance к стенам и занятым клеткам."""
        from scipy.ndimage import distance_transform_edt

        free = ~self.occ
        padded = np.pad(free, 1, constant_values=False)
        dist = distance_transform_edt(padded)[1:-1, 1:-1] * RES
        jj, ii = np.nonzero(dist >= clearance)
        return np.c_[self.x0 + (ii + 0.5) * RES, self.y0 + (jj + 0.5) * RES]


def wall_rect(wall: Wall, s_a: float, s_b: float, off_a: float, off_b: float):
    """Осевой прямоугольник участка стены: s - вдоль оси, off - по левой нормали."""
    pts = np.array([wall.point(s, o) for s in (s_a, s_b) for o in (off_a, off_b)])
    return (*pts.min(0), *pts.max(0))


def room_side(wall: Wall, room: Room) -> int:
    """С какой стороны стены (по левой нормали) лежит помещение."""
    x0, y0, x1, y1 = room.cell
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return 1 if float(np.dot(c - np.asarray(wall.p0), wall.n)) > 0 else -1


def _box_on_side(clear, side: str, pos: float, w: float, d: float, z0: float, h: float,
                 gap: float = 0.02):
    """Прямоугольник и Box предмета, стоящего спиной к стороне чистого прямоугольника."""
    x0, y0, x1, y1 = clear
    if side == "B":
        rect, yaw = (pos, y0 + gap, pos + w, y0 + gap + d), 0.0
    elif side == "T":
        rect, yaw = (pos, y1 - gap - d, pos + w, y1 - gap), math.pi
    elif side == "L":
        rect, yaw = (x0 + gap, pos, x0 + gap + d, pos + w), -math.pi / 2
    else:
        rect, yaw = (x1 - gap - d, pos, x1 - gap, pos + w), math.pi / 2
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    return rect, Box((cx, cy, z0 + h / 2), (w, d, h), yaw)


def _side_len(clear, side: str) -> tuple[float, float]:
    x0, y0, x1, y1 = clear
    return (x0, x1) if side in "BT" else (y0, y1)


def _local_box(base: Box, du: float, dv: float, z0: float, size) -> Box:
    """Box в системе предмета: du - вдоль, dv - от стены (+ к комнате)."""
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    x = base.center[0] + du * c - dv * s
    y = base.center[1] + du * s + dv * c
    return Box((x, y, z0 + size[2] / 2), tuple(size), base.yaw)


# Каталог предметов у стены: (вид, ширина, глубина, высота, где ставить)
WALL_ITEMS = {
    "wardrobe": ((0.8, 2.4), (0.45, 0.65), (1.8, 2.4), ("room", "corridor")),
    "shelf": ((0.6, 1.2), (0.25, 0.4), (1.6, 2.2), ("room",)),
    "sofa": ((1.6, 2.4), (0.8, 1.0), (0.75, 0.9), ("room",)),
    "bed": ((0.9, 1.8), (1.9, 2.1), (0.45, 0.55), ("room",)),
    "desk": ((1.0, 1.6), (0.6, 0.8), (0.73, 0.77), ("room",)),
    "tv_stand": ((1.2, 2.0), (0.35, 0.45), (0.4, 0.6), ("room",)),
    "fridge": ((0.6, 0.7), (0.6, 0.7), (1.7, 2.0), ("kitchen",)),
    "counter": ((1.5, 3.5), (0.6, 0.6), (0.88, 0.92), ("kitchen",)),
    "bathtub": ((1.5, 1.8), (0.7, 0.8), (0.55, 0.6), ("bath",)),
    "sink": ((0.5, 0.8), (0.45, 0.5), (0.8, 0.9), ("bath", "kitchen")),
    "toilet": ((0.36, 0.4), (0.6, 0.7), (0.75, 0.8), ("bath",)),
    "shoe_rack": ((0.6, 1.2), (0.3, 0.4), (0.4, 1.0), ("corridor",)),
}
MANDATORY = {"kitchen": ("counter", "fridge"), "bath": ("bathtub", "sink", "toilet")}


class Furnisher:
    def __init__(self, cfg: InteriorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _u(self, r) -> float:
        return float(self.rng.uniform(r[0], r[1]))

    def _refl(self) -> float:
        if self.rng.random() < self.cfg.p_dark_furniture:
            return self._u((0.03, 0.1))
        return self._u((0.15, 0.8))

    def _add(self, layout: Layout, kind: str, label: str, boxes, refl: float, room_id) -> None:
        layout.items.append(Item(len(layout.items), kind, label, list(boxes), float(refl), room_id))

    # ------------------------------------------------------------------
    def furnish(self, layout: Layout) -> dict[int, RoomGrid]:
        rng = self.rng
        layout.materials = {
            "wall": self._u((0.5, 0.9)), "floor": self._u((0.15, 0.6)),
            "ceiling": self._u((0.7, 0.9)), "door": self._u((0.2, 0.8)),
            "window": 0.8, "glass": 0.05, "mirror": 0.9, "exterior": self._u((0.1, 0.4)),
        }
        self._set_doors(layout)
        grids = {r.id: RoomGrid(r) for r in layout.rooms}
        self._reserve_openings(layout, grids)
        for room in layout.rooms:
            g = grids[room.id]
            self._wall_fixtures(layout, room, g)
            self._columns(layout, room, g)
            if self.cfg.furniture:
                self._furniture(layout, room, g)
            self._clutter(layout, room, g)
        if self.cfg.baseboards:
            for room in layout.rooms:
                self._baseboards(layout, room)
        if self.cfg.exterior_ground:
            self._exterior(layout)
        return grids

    def _set_doors(self, layout: Layout) -> None:
        for o in layout.openings:
            if o.kind != "door":
                continue
            if self.rng.random() < self.cfg.p_closed_door:
                o.angle_deg = 0.0
            else:
                o.angle_deg = round(self._u(self.cfg.door_open_deg), 1)

    def _reserve_openings(self, layout: Layout, grids) -> None:
        for o in layout.openings:
            wall = layout.wall(o.wall_id)
            for rid in o.rooms:
                room = layout.rooms[rid]
                side = room_side(wall, room)
                face = side * wall.thickness / 2
                if o.kind == "window":
                    rect = wall_rect(wall, o.s - o.width / 2 - 0.1, o.s + o.width / 2 + 0.1,
                                     face, face + side * 0.6)
                    grids[rid].mark(rect, 0.0, "no_tall")
                else:
                    depth = max(o.width, 0.9)
                    rect = wall_rect(wall, o.s - o.width / 2 - 0.1, o.s + o.width / 2 + 0.1,
                                     face, face + side * depth)
                    grids[rid].mark(rect, 0.0)

    # --- инженерные элементы на стенах ----------------------------------
    def _wall_fixtures(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        cfg, rng = self.cfg, self.rng
        if cfg.radiators:
            for o in layout.openings:
                if o.kind != "window" or room.id not in o.rooms or o.z0 < 0.45:
                    continue
                wall = layout.wall(o.wall_id)
                side = room_side(wall, room)
                face = side * wall.thickness / 2
                w = min(o.width, 1.2)
                h = min(0.5, o.z0 - 0.15)
                center = wall.point(o.s, face + side * 0.08)
                box = Box((*center, 0.1 + h / 2), (w, 0.08, h), wall.yaw)
                self._add(layout, "radiator", "fixture", [box], self._u((0.6, 0.9)), room.id)
                g.mark(wall_rect(wall, o.s - w / 2, o.s + w / 2, face, face + side * 0.15), 0.0)

        sides = ["B", "T", "L", "R"]
        if rng.random() < cfg.p_soffit:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            d, h = self._u((0.2, 0.5)), self._u((0.2, 0.4))
            _, box = _box_on_side(room.clear, side, a, b - a, d, room.ceiling_z - h, h, gap=0.0)
            self._add(layout, "soffit", "fixture", [box], layout.materials["ceiling"], room.id)

        if rng.random() < cfg.p_pilaster:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            w, d = self._u((0.3, 0.5)), self._u((0.1, 0.25))
            if b - a > w + 0.4:
                pos = self._u((a + 0.2, b - 0.2 - w))
                rect, box = _box_on_side(room.clear, side, pos, w, d, 0.0, layout.ceiling_height,
                                         gap=-0.01)
                if not g.occ[g._slices(rect)].any():
                    self._add(layout, "pilaster", "column", [box], layout.materials["wall"],
                              room.id)
                    g.mark(rect)

        p_mirror = cfg.p_mirror_bath if room.kind == "bath" else cfg.p_mirror_other
        if rng.random() < p_mirror:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            w, h = self._u((0.5, 0.8)), self._u((0.6, 0.9))
            if b - a > w + 0.2:
                pos = self._u((a + 0.1, b - 0.1 - w))
                z0 = self._u((1.0, 1.2))
                rect, box = _box_on_side(room.clear, side, pos, w, 0.005, z0, h, gap=0.003)
                if not g.occ[g._slices(rect)].any():
                    self._add(layout, "mirror", "mirror", [box], layout.materials["mirror"],
                              room.id)
                    g.mark(rect, 0.4, "no_tall")

    def _columns(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        if room.area < self.cfg.column_min_area_m2 or self.rng.random() >= self.cfg.p_column:
            return
        pts = g.free_points(0.8)
        if len(pts) == 0:
            return
        x, y = pts[int(self.rng.integers(len(pts)))]
        s = self._u((0.3, 0.5))
        rect = (x - s / 2, y - s / 2, x + s / 2, y + s / 2)
        box = Box((x, y, layout.ceiling_height / 2), (s, s, layout.ceiling_height), 0.0)
        self._add(layout, "column", "column", [box], layout.materials["wall"], room.id)
        g.mark(rect, 0.3)

    # --- мебель ----------------------------------------------------------
    def _furniture(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        rng = self.rng
        kinds = list(MANDATORY.get(room.kind, ()))
        n_extra = int(rng.poisson(room.area * self.cfg.furniture_per_m2))
        allowed = [k for k, v in WALL_ITEMS.items() if room.kind in v[3]]
        if room.kind == "kitchen":
            allowed += [k for k, v in WALL_ITEMS.items() if "room" in v[3] and k != "bed"]
        if allowed:
            kinds += [allowed[int(rng.integers(len(allowed)))] for _ in range(n_extra)]
        for kind in kinds:
            self._place_wall_item(layout, room, g, kind)
        if room.kind in ("room", "kitchen") and room.area > 8 and rng.random() < 0.6:
            self._place_table(layout, room, g)

    def _place_wall_item(self, layout: Layout, room: Room, g: RoomGrid, kind: str) -> bool:
        rng = self.rng
        wr, dr, hr, _ = WALL_ITEMS[kind]
        w, d, h = self._u(wr), self._u(dr), self._u(hr)
        if kind == "wardrobe" and rng.random() < self.cfg.p_wardrobe_to_ceiling:
            h = room.ceiling_z - 0.005
        h = min(h, room.ceiling_z - 0.02)
        tall = h > 1.0
        refl = self._refl()
        for _ in range(15):
            side = ("B", "T", "L", "R")[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            if b - a < w + 0.04:
                continue
            pos = self._u((a + 0.02, b - 0.02 - w))
            rect, base = _box_on_side(room.clear, side, pos, w, d, 0.0, h)
            if not g.is_free(rect, tall):
                continue
            boxes = self._compose(kind, base, w, d, h)
            self._add(layout, kind, "furniture", boxes, refl, room.id)
            g.mark(rect)
            if kind == "counter" and room.ceiling_z > 2.3:
                upper = _local_box(base, 0.0, -d / 2 + 0.175, 1.45, (w, 0.35, 0.7))
                self._add(layout, "upper_cabinet", "furniture", [upper], refl, room.id)
            return True
        return False

    def _compose(self, kind: str, base: Box, w: float, d: float, h: float) -> list[Box]:
        """Составные предметы - несколько параллелепипедов."""
        if kind == "sofa":
            seat = _local_box(base, 0, 0.1, 0.0, (w, d - 0.2, 0.45))
            back = _local_box(base, 0, -d / 2 + 0.1, 0.0, (w, 0.2, h))
            return [seat, back]
        if kind == "bed":
            mattress = _local_box(base, 0, 0, 0.0, (w, d, h))
            head = _local_box(base, 0, -d / 2 + 0.03, 0.0, (w, 0.06, 1.0))
            return [mattress, head]
        if kind == "desk":
            return _table_boxes(base, w, d, h)
        if kind == "toilet":
            bowl = _local_box(base, 0, 0.08, 0.0, (w, d - 0.16, 0.42))
            tank = _local_box(base, 0, -d / 2 + 0.09, 0.0, (w, 0.18, h))
            return [bowl, tank]
        return [base]

    def _place_table(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        rng = self.rng
        w, d, h = self._u((1.0, 1.8)), self._u((0.7, 1.0)), self._u((0.72, 0.76))
        pts = g.free_points(max(w, d) / 2 + 0.5)
        if len(pts) == 0:
            return
        x, y = pts[int(rng.integers(len(pts)))]
        yaw = float(rng.uniform(-0.3, 0.3)) + (math.pi / 2 if rng.random() < 0.5 else 0.0)
        base = Box((x, y, h / 2), (w, d, h), yaw)
        boxes = _table_boxes(base, w, d, h)
        for k in range(int(rng.integers(0, 5))):
            du = (k % 2 * 2 - 1) * w / 4
            dv = (1 if k < 2 else -1) * (d / 2 + 0.3)
            seat = _local_box(base, du, dv, 0.0, (0.45, 0.45, 0.45))
            back_dv = dv + np.sign(dv) * 0.2
            back = _local_box(base, du, back_dv, 0.0, (0.45, 0.05, 0.9))
            boxes += [seat, back]
        self._add(layout, "dining_table", "furniture", boxes, self._refl(), room.id)
        r = max(w, d) / 2 + 0.6
        g.mark((x - r, y - r, x + r, y + r), 0.0)

    def _clutter(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        n = int(self.rng.poisson(room.area * self.cfg.clutter_per_m2))
        for _ in range(n):
            s = self._u((0.25, 0.6))
            pts = g.free_points(s)
            if len(pts) == 0:
                return
            x, y = pts[int(self.rng.integers(len(pts)))]
            h = self._u((0.2, 0.7))
            box = Box((x, y, h / 2), (s, self._u((0.25, 0.6)), h), float(self.rng.uniform(0, math.pi)))
            self._add(layout, "box", "clutter", [box], self._refl(), room.id)
            g.mark((x - s / 2, y - s / 2, x + s / 2, y + s / 2), 0.1)

    # --- плинтусы ----------------------------------------------------------
    def _baseboards(self, layout: Layout, room: Room) -> None:
        x0, y0, x1, y1 = room.clear
        h, d = 0.07, 0.015
        refl = self._u((0.3, 0.9))
        boxes = []
        for side in ("B", "T", "L", "R"):
            a, b = _side_len(room.clear, side)
            coord = {"B": y0, "T": y1, "L": x0, "R": x1}[side]
            gaps = []
            for o in layout.openings:
                if o.z0 > 0.05:
                    continue
                wall = layout.wall(o.wall_id)
                vertical_wall = abs(wall.u[0]) < 0.5
                if vertical_wall != (side in "LR"):
                    continue
                line = wall.p0[0] if vertical_wall else wall.p0[1]
                if abs(line - coord) > wall.thickness / 2 + 0.02:
                    continue
                c = wall.point(o.s)[1 if vertical_wall else 0]
                gaps.append((c - o.width / 2, c + o.width / 2))
            for sa, sb in _subtract_intervals((a, b), gaps):
                if sb - sa < 0.1:
                    continue
                _, box = _box_on_side(room.clear, side, sa, sb - sa, d, 0.0, h, gap=0.0)
                boxes.append(box)
        if boxes:
            self._add(layout, "baseboard", "fixture", boxes, refl, room.id)

    # --- окружение за окнами -------------------------------------------
    def _exterior(self, layout: Layout) -> None:
        rng = self.rng
        x0, y0, x1, y1 = layout.bbox()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        drop = self._u(self.cfg.ground_drop_m)
        refl = layout.materials["exterior"]
        ground = Box((cx, cy, -drop - 0.35), (160.0, 160.0, 0.2), 0.0)
        self._add(layout, "ground", "exterior", [ground], refl, None)
        lo, hi = self.cfg.neighbour_buildings
        for _ in range(int(rng.integers(lo, hi + 1))):
            ang = rng.uniform(0, 2 * math.pi)
            sx, sy, sz = rng.uniform(8, 30), rng.uniform(8, 30), rng.uniform(10, 40)
            dist = rng.uniform(15, 40) + math.hypot(x1 - x0, y1 - y0) / 2 + math.hypot(sx, sy) / 2
            c = (cx + dist * math.cos(ang), cy + dist * math.sin(ang), -drop - 0.25 + sz / 2)
            self._add(layout, "building", "exterior", [Box(c, (sx, sy, sz), float(ang))],
                      self._u((0.1, 0.6)), None)


def _table_boxes(base: Box, w: float, d: float, h: float) -> list[Box]:
    top = _local_box(base, 0, 0, h - 0.03, (w, d, 0.03))
    legs = [_local_box(base, su * (w / 2 - 0.05), sv * (d / 2 - 0.05), 0.0, (0.05, 0.05, h - 0.03))
            for su in (-1, 1) for sv in (-1, 1)]
    return [top, *legs]


def _subtract_intervals(span, gaps):
    a, b = span
    out = []
    cur = a
    for ga, gb in sorted(gaps):
        if gb <= cur or ga >= b:
            continue
        if ga > cur:
            out.append((cur, ga))
        cur = max(cur, gb)
    if cur < b:
        out.append((cur, b))
    return out
