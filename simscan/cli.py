"""Командная строка.

    python -m simscan generate --out D:/synth --count 200 --seed 0 [--config cfg.yaml] [--workers 2]
    python -m simscan generate --out D:/synth --count 1 --set scanner.angular_step_deg=0.3
    python -m simscan dump-config > cfg.yaml
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
    args = ap.parse_args(argv)

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
