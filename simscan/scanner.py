"""Симулятор наземного лазерного сканера на Open3D RaycastingScene.

Станция пускает регулярную сетку лучей (азимут x угол места), как настоящий TLS;
номера строки и столбца пишутся в E57 (rowIndex / columnIndex).
Эффекты (каждый под своим флагом в EffectsConfig):
  - шум дальности, зависящий от дальности и отражательной способности;
  - смешанные пиксели на перепадах глубины («хвосты» на кромках);
  - пропуски при скользящем падении и при слабом сигнале (тёмные материалы, даль);
  - стекло: луч проходит насквозь с вероятностью p, иначе редкий отклик от стекла;
  - зеркало: отражённый луч, точка на суммарной дальности за зеркалом (virtual = True);
  - люди и временные предметы - свои на каждой станции;
  - лучи без отклика сохраняются как направления (cartesianInvalidState = 1).
Ошибка регистрации и поворот в СК объекта применяются при записи (generate.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import EffectsConfig, ScannerConfig
from .labels import LABEL_ID
from .layout import Box, Layout
from .scene import SceneMesh, Solid
from .transform import rot_x, rot_y, rot_z

GLASS, MIRROR = LABEL_ID["glass"], LABEL_ID["mirror"]


@dataclass
class Station:
    id: int
    room_id: int
    position: tuple[float, float, float]     # в СК планировки
    heading_deg: float                       # куда смотрит нулевой азимут сканера
    tilt_deg: tuple[float, float] = (0.0, 0.0)

    def rotation(self) -> np.ndarray:
        """Поворот «СК сканера -> СК планировки»."""
        tx, ty = (math.radians(a) for a in self.tilt_deg)
        return rot_z(math.radians(self.heading_deg)) @ rot_y(ty) @ rot_x(tx)


@dataclass
class ScanResult:
    station_id: int
    nrow: int
    ncol: int
    xyz_local: np.ndarray      # (N, 3) float64; для лучей без отклика - направление * 0,1 м
    valid: np.ndarray          # (N,) bool
    row: np.ndarray            # (N,) uint16
    col: np.ndarray            # (N,) uint16
    intensity: np.ndarray      # (N,) float32, 0..1
    label: np.ndarray          # (N,) uint8, см. labels.py
    instance: np.ndarray       # (N,) int32
    virtual: np.ndarray        # (N,) bool - точка из отражения в зеркале
    mixed: np.ndarray          # (N,) bool - смешанный пиксель

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def ray_grid(cfg: ScannerConfig):
    """Направления лучей в СК сканера: строки сверху вниз, столбцы по азимуту 0..360."""
    step = cfg.angular_step_deg
    ncol = int(round(360.0 / step))
    el_top = cfg.elevation_max_deg - (step / 2 if cfg.elevation_max_deg >= 90.0 else 0.0)
    nrow = int(math.floor((el_top - cfg.elevation_min_deg) / step + 1e-9)) + 1
    if nrow > 65535 or ncol > 65535:
        raise ValueError("сетка не помещается в uint16 rowIndex/columnIndex")
    el = np.radians(el_top - np.arange(nrow) * step)
    az = np.arange(ncol) * (2 * math.pi / ncol)
    ce = np.cos(el)[:, None]
    dirs = np.stack([ce * np.cos(az)[None], ce * np.sin(az)[None],
                     np.broadcast_to(np.sin(el)[:, None], (nrow, ncol))], -1)
    rows = np.repeat(np.arange(nrow, dtype=np.uint16), ncol)
    cols = np.tile(np.arange(ncol, dtype=np.uint16), nrow)
    return dirs.reshape(-1, 3), rows, cols, nrow, ncol


def place_stations(layout: Layout, grids: dict, cfg: ScannerConfig,
                   rng: np.random.Generator) -> list[Station]:
    stations: list[Station] = []
    for room in layout.rooms:
        if len(layout.rooms) > 1 and rng.random() < cfg.p_room_unscanned:
            room.scanned = False
            continue
        g = grids[room.id]
        pts = g.free_points(cfg.wall_clearance_m)
        if len(pts) == 0:
            pts = g.free_points(0.3)
        if len(pts) == 0:
            room.scanned = False
            continue
        n = int(np.clip(round(room.area * cfg.stations_per_m2 + rng.uniform(-0.5, 0.5)),
                        1, cfg.max_stations_per_room))
        chosen = [pts[int(rng.integers(len(pts)))]]
        for _ in range(n - 1):
            d = np.min(np.linalg.norm(pts[:, None] - np.array(chosen)[None], axis=2), axis=1)
            if d.max() < 1.0:
                break
            chosen.append(pts[int(np.argmax(d))])
        for x, y in chosen:
            z = float(rng.uniform(*cfg.height_m))
            tilt = tuple(float(v) for v in rng.normal(0, cfg.tilt_sigma_deg, 2)) \
                if cfg.tilt_sigma_deg > 0 else (0.0, 0.0)
            stations.append(Station(len(stations), room.id, (float(x), float(y), z),
                                    float(rng.uniform(0, 360)), tilt))
    return stations


def people_for_station(station: Station, grids: dict, n: int, rng: np.random.Generator,
                       instance0: int) -> list[Solid]:
    """Люди стоят в том же помещении, что и станция; на каждой станции - свои."""
    pts = grids[station.room_id].free_points(0.3)
    if len(pts) == 0 or n == 0:
        return []
    far = pts[np.linalg.norm(pts - np.array(station.position[:2]), axis=1) > 0.8]
    if len(far) == 0:
        return []
    out = []
    for k in range(n):
        x, y = far[int(rng.integers(len(far)))]
        h = float(rng.uniform(1.55, 1.9))
        box = Box((float(x), float(y), h / 2), (0.45, 0.28, h), float(rng.uniform(0, math.pi)))
        out.append(Solid(box, "clutter", instance0 + k, float(rng.uniform(0.1, 0.5)), "person"))
    return out


def sample_effect_params(ecfg: EffectsConfig, rng: np.random.Generator) -> dict:
    """Параметры эффектов на сцену (рандомизация домена); пишутся в layout.json."""
    u = lambda r: float(rng.uniform(r[0], r[1]))  # noqa: E731
    return {
        "range_sigma_m": u(ecfg.range_sigma_mm) / 1000 if ecfg.range_noise else 0.0,
        "range_sigma_per_m": u(ecfg.range_sigma_per_m_mm) / 1000 if ecfg.range_noise else 0.0,
        "p_mixed": u(ecfg.p_mixed) if ecfg.mixed_pixels else 0.0,
        "p_glass_pass": u(ecfg.p_glass_pass) if ecfg.glass else 0.0,
        "registration_sigma_m": u(ecfg.registration_sigma_mm) / 1000 if ecfg.registration_error else 0.0,
        "registration_sigma_deg": u(ecfg.registration_sigma_deg) if ecfg.registration_error else 0.0,
    }


class ScanSimulator:
    def __init__(self, mesh: SceneMesh, scfg: ScannerConfig, ecfg: EffectsConfig,
                 params: dict, rng: np.random.Generator):
        self.mesh = mesh
        self.scfg, self.ecfg, self.params, self.rng = scfg, ecfg, params, rng
        self.dirs, self.rows, self.cols, self.nrow, self.ncol = ray_grid(scfg)
        self._static_scene = mesh.to_raycasting_scene()

    # ------------------------------------------------------------------
    def _cast(self, scene, origins: np.ndarray, dirs: np.ndarray):
        import open3d as o3d

        n = len(dirs)
        t = np.empty(n, np.float64)
        prim = np.empty(n, np.int64)
        nrm = np.empty((n, 3), np.float64)
        step = self.scfg.chunk_rays
        for a in range(0, n, step):
            b = min(a + step, n)
            rays = np.empty((b - a, 6), np.float32)
            rays[:, :3] = origins[a:b] if origins.ndim == 2 else origins
            rays[:, 3:] = dirs[a:b]
            res = scene.cast_rays(o3d.core.Tensor(rays))
            t[a:b] = res["t_hit"].numpy()
            prim[a:b] = res["primitive_ids"].numpy().astype(np.int64)
            nrm[a:b] = res["primitive_normals"].numpy()
        hit = np.isfinite(t)
        prim[~hit] = -1
        return t, prim, nrm, hit

    def scan(self, station: Station, dynamic: list[Solid] | None = None) -> ScanResult:
        ecfg, p, rng = self.ecfg, self.params, self.rng
        mesh, scene = self.mesh, self._static_scene
        if dynamic:
            mesh = mesh.concatenate(SceneMesh.from_solids(dynamic))
            scene = mesh.to_raycasting_scene()

        R = station.rotation()
        o = np.asarray(station.position, float)
        d = self.dirs @ R.T
        t, prim, nrm, hit = self._cast(scene, o, d)
        n = len(t)
        virtual = np.zeros(n, bool)
        via_special = np.zeros(n, bool)    # луч задел стекло или зеркало
        dir_eff = d.copy()                 # направление последнего сегмента луча (для угла падения)

        def lookup(sel):
            return (mesh.tri_label[prim[sel]], mesh.tri_instance[prim[sel]],
                    mesh.tri_reflectance[prim[sel]])

        label = np.zeros(n, np.uint8)
        inst = np.zeros(n, np.int32)
        refl = np.zeros(n, np.float32)
        label[hit], inst[hit], refl[hit] = lookup(hit)

        # --- стекло и зеркала в любом порядке вдоль луча ---------------------
        # t - полный путь луча; точка ставится на исходном направлении на дальности t
        # (так сканер и видит отражение: «виртуальная» точка за зеркалом).
        if ecfg.glass or ecfg.mirrors:
            seg_o = np.broadcast_to(o, (n, 3)).copy()   # начало текущего отрезка луча
            seg_t = t.copy()                              # длина текущего отрезка
            settled = np.zeros(n, bool)                   # луч остался на стекле
            # стекло даёт два пересечения (две грани), на луче бывает несколько окон и зеркал
            for _ in range(8):
                g = np.nonzero(hit & (label == GLASS) & ~settled)[0] if ecfg.glass else np.empty(0, int)
                m = np.nonzero(hit & (label == MIRROR))[0] if ecfg.mirrors else np.empty(0, int)
                if len(g):
                    via_special[g] = True
                    through = rng.random(len(g)) < p["p_glass_pass"]
                    stop = g[~through]
                    settled[stop] = True
                    hit[stop[rng.random(len(stop)) >= ecfg.p_glass_return]] = False
                    g = g[through]
                if len(g) == 0 and len(m) == 0:
                    break
                idx = np.concatenate([g, m])
                new_dir = dir_eff[idx].copy()
                if len(m):
                    nm = nrm[m] / np.linalg.norm(nrm[m], axis=1, keepdims=True)
                    dm = dir_eff[m]
                    new_dir[len(g):] = dm - 2 * np.sum(dm * nm, axis=1, keepdims=True) * nm
                    virtual[m] = True
                    via_special[m] = True
                o2 = seg_o[idx] + dir_eff[idx] * seg_t[idx, None] + new_dir * 1e-3
                t2, prim2, nrm2, hit2 = self._cast(scene, o2, new_dir)
                t[idx] = t[idx] + 1e-3 + t2
                seg_o[idx], seg_t[idx], dir_eff[idx] = o2, t2, new_dir
                prim[idx], nrm[idx], hit[idx] = prim2, nrm2, hit2
                label[idx] = 0
                hi = idx[hit2]
                label[hi], inst[hi], refl[hi] = lookup(hi)
            # лимит итераций исчерпан - отклика нет
            hit[hit & (((label == GLASS) & ~settled) | ((label == MIRROR) & ecfg.mirrors))] = False
            refl[virtual] *= 0.9

        # --- рабочий диапазон дальностей ------------------------------------
        hit &= (t >= self.scfg.min_range_m) & (t <= self.scfg.max_range_m)

        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        cos_inc = np.abs(np.sum(dir_eff * nrm, axis=1))
        cos_inc[~hit] = 0.0

        # --- пропуски -------------------------------------------------------
        if ecfg.grazing_dropout:
            ang = np.degrees(np.arccos(np.clip(cos_inc, 0, 1)))
            span = ecfg.grazing_full_deg - ecfg.grazing_start_deg
            p_drop = np.clip((ang - ecfg.grazing_start_deg) / span, 0, 1) ** 2
            hit &= rng.random(n) >= p_drop
        if ecfg.low_signal_dropout:
            snr = refl * np.maximum(cos_inc, 0.05) / np.maximum(t / 10.0, 0.1) ** 2
            p_drop = np.exp(-snr / ecfg.low_signal_snr0)
            hit &= rng.random(n) >= p_drop

        # --- смешанные пиксели ----------------------------------------------
        mixed = np.zeros(n, bool)
        if ecfg.mixed_pixels and p["p_mixed"] > 0:
            # у стекла «перепад» между соседями - случайность прохода, а не кромка
            mixed = self._mixed_pixels(t, hit & ~via_special, p["p_mixed"])

        # --- шум дальности --------------------------------------------------
        if ecfg.range_noise and p["range_sigma_m"] > 0:
            sig = (p["range_sigma_m"] + p["range_sigma_per_m"] * t) \
                * np.sqrt(0.5 / np.maximum(refl, 0.05))
            sig = np.minimum(sig, 4 * p["range_sigma_m"] + p["range_sigma_per_m"] * t)
            noise = rng.normal(0.0, 1.0, n) * np.where(hit, sig, 0.0)
            t = np.where(hit, t + noise, t)

        # --- результат ------------------------------------------------------
        tt = np.where(hit, t, 0.1)
        xyz = self.dirs * tt[:, None]
        intensity = np.where(
            hit, refl * np.power(cos_inc, 0.7) * np.exp(-np.where(hit, t, 0) / 80.0)
            + rng.normal(0, 0.01, n), 0.0)
        label[~hit], inst[~hit] = 0, 0
        return ScanResult(station.id, self.nrow, self.ncol, xyz, hit, self.rows, self.cols,
                          np.clip(intensity, 0, 1).astype(np.float32), label, inst,
                          virtual & hit, mixed & hit)

    def _mixed_pixels(self, t: np.ndarray, hit: np.ndarray, p_mixed: float) -> np.ndarray:
        """Смешанный пиксель: дальность между передним и задним планом у кромки."""
        T = np.where(hit, t, np.nan).reshape(self.nrow, self.ncol)
        jump = self.ecfg.mixed_jump_m
        best = np.full(T.shape, np.nan)
        neighbours = [np.roll(T, -1, 1), np.roll(T, 1, 1)]
        down = np.full(T.shape, np.nan)
        down[:-1] = T[1:]
        up = np.full(T.shape, np.nan)
        up[1:] = T[:-1]
        neighbours += [down, up]
        for nb in neighbours:
            cand = np.isnan(best) & np.isfinite(T) & np.isfinite(nb) & (np.abs(T - nb) > jump)
            best[cand] = nb[cand]
        cand = np.isfinite(best)
        pick = cand & (self.rng.random(T.shape) < p_mixed)
        u = self.rng.uniform(0.1, 0.9, T.shape)
        T2 = np.where(pick, T + u * (best - T), T)
        flat = T2.reshape(-1)
        t[pick.reshape(-1)] = flat[pick.reshape(-1)]
        return pick.reshape(-1)
