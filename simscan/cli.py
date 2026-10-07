"""Командная строка.

    python -m simscan generate --out D:/synth --count 200 --seed 0 [--config cfg.yaml] [--workers 2]
    python -m simscan generate --out D:/synth --count 1 --set scanner.angular_step_deg=0.3
    python -m simscan dump-config > cfg.yaml
    python -m simscan inspect-e57 D:/scans/kvartira.e57    # отчёт по реальному E57 -> _report.zip
    python -m simscan assets --source gso --out D:/scans --per-role 40
    python -m simscan pair D:/synth/scene_00000 D:/synth/scene_00001 --out pairs.png
"""
from __future__ import annotations

import argparse
import sys

# тяжёлые зависимости (Open3D, pye57) импортируются внутри команд: inspect-e57 работает без них


def _parse_set(items: list[str]) -> dict:
    """--set a.b=1 --set c.d=[1,2] -> вложенный словарь (значения разбираются как YAML)."""
    import yaml

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

    from .config import load_config

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
    ie = sub.add_parser("inspect-e57", help="отчёт по реальному E57 для калибровки (маленький zip)")
    ie.add_argument("e57")
    ie.add_argument("--out", help="каталог отчёта (по умолчанию <файл>_report рядом с файлом)")
    ie.add_argument("--no-sample", action="store_true", help="без прореженного облака sample.npz")
    ie.add_argument("--chunk", type=int, default=2_000_000, help="точек за одно чтение")
    pr = sub.add_parser("pair", help="обучающая пара сцены: растр свободного пространства + эталон")
    pr.add_argument("scene_dirs", nargs="+")
    pr.add_argument("--out", help="общая картинка (по умолчанию <сцена>/input/pair.png)")
    args = ap.parse_args(argv)

    if args.cmd == "assets":
        from . import assets

        fn = {"gso": assets.fetch_gso, "polyhaven": assets.fetch_polyhaven,
              "objaverse": assets.fetch_objaverse}[args.source]
        kw = {"per_role": args.per_role, "seed": args.seed} if args.source == "gso" \
            else {"max_per_kind": args.per_role}
        print(fn(args.out, max_triangles=args.max_triangles, **kw))
        return
    if args.cmd == "inspect-e57":
        from pathlib import Path

        from .inspect_e57 import inspect_e57

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_report")
        rep = inspect_e57(src, out, chunk=args.chunk, sample=not args.no_sample)
        print((out / "summary.txt").read_text(encoding="utf-8"))
        print("отправить:", rep.get("zip"))
        return
    if args.cmd == "pair":
        import json
        from pathlib import Path

        from .rasterize import pair_data, pair_sheet

        items = [pair_data(d) for d in args.scene_dirs]
        out = Path(args.out) if args.out else Path(args.scene_dirs[0]) / "input" / "pair.png"
        pair_sheet(items, out)
        print(json.dumps({d["name"]: {k: v for k, v in d["stats"].items() if k.startswith("wall")}
                          for d in items}, ensure_ascii=False, indent=1))
        print(out)
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
        import yaml

        from .config import config_to_dict, load_config

        cfg = load_config(args.config)
        yaml.safe_dump(config_to_dict(cfg), sys.stdout, allow_unicode=True, sort_keys=False)
        return
    from .config import load_config
    from .generate import generate_dataset

    cfg = load_config(args.config, _parse_set(args.set))
    results = generate_dataset(cfg, args.out, args.count, args.seed, args.start, args.workers)
    failed = [r for r in results if "error" in r]
    if failed:
        print(f"ошибок: {len(failed)} из {len(results)}", file=sys.stderr)
        sys.exit(1)
