import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simscan.config import load_config  # noqa: E402

# всё «чистое»: без шума и помех, грубая сетка лучей
CLEAN_EFFECTS = {
    "range_noise": False, "mixed_pixels": False, "grazing_dropout": False,
    "low_signal_dropout": False, "glass": False, "mirrors": False,
    "registration_error": False, "people": False,
}


def box_room_config(w=5.0, d=3.0, **extra):
    """Одна комната w x d (по внутренним граням), без проёмов, мебели и эффектов."""
    over = {
        "layout": {"footprint_x_m": [w, w], "footprint_y_m": [d, d], "rooms": [1, 1],
                   "p_l_shape": 0.0, "p_entrance": 0.0, "p_window_per_side": 0.0,
                   "p_suspended_ceiling": 0.0},
        "interior": {"furniture": False, "furniture_per_m2": 0.0, "clutter_per_m2": 0.0, "radiators": False,
                     "baseboards": False, "p_soffit": 0.0, "p_column": 0.0, "p_pilaster": 0.0,
                     "p_mirror_bath": 0.0, "p_mirror_other": 0.0, "exterior_ground": False},
        "scanner": {"angular_step_deg": 0.5, "p_room_unscanned": 0.0, "max_stations_per_room": 1},
        "effects": dict(CLEAN_EFFECTS),
        "export": {"preview": False},
    }
    for k, v in extra.items():
        over.setdefault(k, {}).update(v)
    return load_config(overrides=over)


@pytest.fixture
def coarse_config():
    return load_config(overrides={"scanner": {"angular_step_deg": 0.5},
                                  "export": {"preview": False}})
