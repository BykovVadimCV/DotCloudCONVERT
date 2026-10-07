"""Командная строка.

    python -m simscan generate --out D:/synth --count 200 --seed 0 [--config cfg.yaml] [--workers 2]
    python -m simscan generate --out D:/synth --count 1 --set scanner.angular_step_deg=0.3
    python -m simscan dump-config > cfg.yaml
    python -m simscan assets --source gso --out D:/scans --per-role 40
    python -m simscan pair D:/synth/scene_00000          # вход сети и эталон рядом
"""
from __future__ import annotations

import argparse
import sys

import yaml

from .config import config_to_dict, load_config
from .generate import generate_dataset


def _parse_set(items: list[str]) -> dict:
    """--set a.b=1 --set c.d=[1,2] -> вложенный словарь (значения разбираются как YAML)."""
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def _plans(args) -> None:
    from pathlib import Path

    from .generate import make_layout, scene_rng
    from .interior import Furnisher
    from .render import render_many, render_plan

    cfg = load_config(args.config, _parse_set(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts = []
    for i in range(args.count):
        rng = scene_rng(args.seed, i)
        lay = make_layout(cfg, rng, args.seed, i)
        Furnisher(cfg.interior, rng).furnish(lay)
        lay.save_json(out / f"plan_{i:04d}.json")
        render_plan(lay, out / f"plan_{i:04d}.png")
        layouts.append(lay)
    render_many(layouts, out / "sheet.png")
    print(out / "sheet.png")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="simscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="сгенерировать сцены")
    g.add_argument("--out", required=True)
    g.add_argument("--count", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--start", type=int, default=0, help="номер первой сцены (для дозапуска)")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--config")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    d = sub.add_parser("dump-config", help="напечатать конфигурацию по умолчанию (YAML)")
    d.add_argument("--config")
    pl = sub.add_parser("plans", help="только планировки + рендер (без сканирования)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--count", type=int, default=6)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--config")
    pl.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r = sub.add_parser("render", help="отрисовать layout.json сцены")
    r.add_argument("scene_dir")
    dbg = sub.add_parser("debug", help="отладочные визуализации сцены в <сцена>/debug")
    dbg.add_argument("scene_dir")
    a = sub.add_parser("assets", help="скачать реальные 3D-модели (сканы предметов, мебель)")
    a.add_argument("--source", choices=("gso", "polyhaven", "objaverse"), default="gso")
    a.add_argument("--out", required=True,
                   help="gso -> interior.scan_dir; polyhaven/objaverse -> interior.asset_dir")
    a.add_argument("--per-role", type=int, default=20)
    a.add_argument("--max-triangles", type=int, default=3000)
    a.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("pair", help="обучающая пара сцены: растр свободного пространства + эталон")
    pr.add_argument("scene_dir")
    args = ap.parse_args(argv)

    if args.cmd == "assets":
        from . import assets

        fn = {"gso": assets.fetch_gso, "polyhaven": assets.fetch_polyhaven,
              "objaverse": assets.fetch_objaverse}[args.source]
        kw = {"per_role": args.per_role, "seed": args.seed} if args.source == "gso" \
            else {"max_per_kind": args.per_role}
        print(fn(args.out, max_triangles=args.max_triangles, **kw))
        return
    if args.cmd == "pair":
        import json

        from .rasterize import make_pair

        print(json.dumps(make_pair(args.scene_dir), ensure_ascii=False, indent=1))
        return

    if args.cmd == "debug":
        import json

        from .debug import debug_scene

        print(json.dumps(debug_scene(args.scene_dir), ensure_ascii=False, indent=1)[:2000])
        return

    if args.cmd == "plans":
        _plans(args)
        return
    if args.cmd == "render":
        from pathlib import Path

        from .layout import Layout
        from .render import render_plan

        d = Path(args.scene_dir)
        render_plan(Layout.load_json(d / "layout.json"), d / "plan.png")
        print(d / "plan.png")
        return

    if args.cmd == "dump-config":
        cfg = load_config(args.config)
        yaml.safe_dump(config_to_dict(cfg), sys.stdout, allow_unicode=True, sort_keys=False)
        return
    cfg = load_config(args.config, _parse_set(args.set))
    results = generate_dataset(cfg, args.out, args.count, args.seed, args.start, args.workers)
    failed = [r for r in results if "error" in r]
    if failed:
        print(f"ошибок: {len(failed)} из {len(results)}", file=sys.stderr)
        sys.exit(1)
