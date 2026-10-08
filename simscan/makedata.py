"""Массовая сборка датасета для сети плана: сцена генератора -> растр -> удаление скана.

    python -m simscan make-dataset --config configs/customer_like.yaml --count 3000 \
        --workers 8 --out dataset
    python -m simscan dataset pack dataset --out dataset.zip       # один архив для Google Drive

Скан E57 сцены весит ~0,5 ГБ, готовый уровень - ~5 МБ: каждый рабочий процесс генерирует
сцену во временный каталог, растеризует её в dataset/scenes/<scene>/L1 и удаляет скан.
Повторный запуск с теми же --seed/--start продолжает с места обрыва (готовые сцены
пропускаются). Несколько машин: одинаковый --seed, разные --start (непересекающиеся номера),
потом каталоги scenes/ слить и выполнить `dataset splits`.

Аугментации облака (до растеризации): прореживание и сдвиг по z - на доле сцен.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import SynthConfig


def _done(root: Path, scene_id: str) -> bool:
    return (root / "scenes" / scene_id / "L1" / "scene.json").exists()


def _aug(seed: int, index: int, p_thin: float, p_shift: float) -> tuple[float, float]:
    rng = np.random.default_rng([seed, index, 7])
    thin = float(rng.uniform(0.2, 0.6)) if rng.random() < p_thin else 0.0
    z_shift = float(rng.normal(0, 0.01)) if rng.random() < p_shift else 0.0
    return round(thin, 3), round(z_shift, 4)


def _one(args) -> dict:
    cfg, root, work, seed, index, keep_scan, p_thin, p_shift = args
    from .generate import generate_scene
    from .netinput import build_scene

    scene_id = f"scene_{index:05d}"
    tmp = Path(work) / scene_id
    t0 = time.time()
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        r = generate_scene(cfg, tmp, seed, index)
        if isinstance(r, dict) and r.get("error"):
            raise RuntimeError(r["error"])
        thin, z_shift = _aug(seed, index, p_thin, p_shift)
        dirs = build_scene(tmp, root, thin=thin, z_shift=z_shift, splits=False, log=lambda *a, **k: None)
        return {"scene": scene_id, "levels": [str(Path(d).relative_to(Path(root) / "scenes")) for d in dirs],
                "s": round(time.time() - t0, 1)}
    except Exception as exc:  # одна плохая сцена не роняет набор
        return {"scene": scene_id, "error": f"{type(exc).__name__}: {exc}", "s": round(time.time() - t0, 1)}
    finally:
        if not keep_scan:
            shutil.rmtree(tmp, ignore_errors=True)


def rebuild_splits(root) -> dict:
    """splits/*.txt заново по всем scenes/*/L*/scene.json (после слияния машин или сбоя)."""
    from .netinput import _split_of

    root = Path(root)
    sp = {"train": [], "val": [], "test": []}
    for f in sorted((root / "scenes").glob("*/L*/scene.json")):
        sc = json.loads(f.read_text(encoding="utf-8"))
        sp[_split_of(sc["layout_key"])].append(f"{f.parent.parent.name}/{f.parent.name}")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for k, v in sp.items():
        (root / "splits" / f"{k}.txt").write_text("".join(x + "\n" for x in v), encoding="utf-8")
    return {k: len(v) for k, v in sp.items()}


def make_dataset(cfg: SynthConfig, out_root, count: int, seed: int = 0, start: int = 0, workers: int = 1,
                 work_dir=None, keep_scans: bool = False, p_thin: float = 0.3, p_shift: float = 0.5,
                 log=print) -> dict:
    from .netinput import write_meta

    root = Path(out_root)
    write_meta(root)
    # лишнее для датасета сети не пишем: превью, маски datasetgen, отладку
    cfg.export.preview = cfg.export.unet_mask = cfg.export.debug = cfg.export.free_space_input = False
    cfg.export.write_mesh = False
    work = Path(work_dir) if work_dir else root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(start, start + count) if not _done(root, f"scene_{i:05d}")]
    log(f"сцен {count}, готово {count - len(todo)}, осталось {len(todo)}")
    jobs = [(cfg, str(root), str(work), seed, i, keep_scans, p_thin, p_shift) for i in todo]
    ok = err = 0
    t0 = time.time()
    with open(root / "make_log.jsonl", "a", encoding="utf-8") as flog:
        def take(r):
            nonlocal ok, err
            flog.write(json.dumps(r, ensure_ascii=False) + "\n")
            flog.flush()
            if r.get("error"):
                err += 1
            else:
                ok += 1
            n = ok + err
            eta = (time.time() - t0) / n * (len(todo) - n)
            log(f"[{n}/{len(todo)}] {r['scene']} {'ОШИБКА ' + r['error'][:120] if r.get('error') else 'ок'}"
                f" ({r['s']} с; осталось ~{eta / 3600:.1f} ч)")

        if workers <= 1:
            for j in jobs:
                take(_one(j))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fut in as_completed([pool.submit(_one, j) for j in jobs]):
                    take(fut.result())
    if not keep_scans:
        shutil.rmtree(work, ignore_errors=True)
    splits = rebuild_splits(root)
    log(f"готово: {ok}, ошибок: {err}; splits {splits}")
    return {"ok": ok, "errors": err, "splits": splits}


def pack(root, out_zip, include_previews: bool = False) -> Path:
    """Датасет -> один zip (deflate; входы и разметка в основном нули и сжимаются в разы)."""
    root, out_zip = Path(root), Path(out_zip)
    skip = {"_work"}
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if f.is_dir() or rel.parts[0] in skip or (f.name == "preview.png" and not include_previews):
                continue
            z.write(f, Path(root.name) / rel)
    return out_zip
