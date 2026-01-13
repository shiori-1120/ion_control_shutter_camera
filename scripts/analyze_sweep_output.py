"""Analyze Sweep output (shots.csv) for dry-mode misclassification.

Usage:
  python scripts/analyze_sweep_output.py <output_dir>
  python scripts/analyze_sweep_output.py <output_dir> --max 30

Where <output_dir> is like:
  data/output/spectrum/20260108_123456

It reads:
  - shots.csv (required)
  - threshold.json (optional)

This script intentionally uses only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Row:
    bright: int
    label_bright: int | None
    s_norm: float | None
    tau_on: float | None
    tau_off: float | None
    cam_sample: str


def _parse_int01(s: str) -> int | None:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        v = int(float(s))
    except ValueError:
        return None
    if v in (0, 1):
        return v
    return None


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_rows(shots_csv: Path) -> list[Row]:
    rows: list[Row] = []
    with shots_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            bright = _parse_int01(str(r.get("bright", "")))
            if bright is None:
                continue
            rows.append(
                Row(
                    bright=bright,
                    label_bright=_parse_int01(str(r.get("label_bright", ""))),
                    s_norm=_parse_float(str(r.get("S_norm", ""))),
                    tau_on=_parse_float(str(r.get("tau_on", ""))),
                    tau_off=_parse_float(str(r.get("tau_off", ""))),
                    cam_sample=str(r.get("cam_sample", "") or ""),
                )
            )
    return rows


def _fmt_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--max", type=int, default=30, help="Max rows to list per category")
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    shots_csv = out_dir / "shots.csv"
    if not shots_csv.exists():
        raise SystemExit(f"shots.csv not found: {shots_csv}")

    threshold_json = out_dir / "threshold.json"
    if threshold_json.exists():
        try:
            th = json.loads(threshold_json.read_text(encoding="utf-8"))
            roi = th.get("roi")
            tau = None
            try:
                tau = th.get("threshold", {}).get("tau")
            except Exception:
                tau = None
            print("=== threshold.json ===")
            print(f"roi: {roi}")
            if tau is not None:
                print(f"tau(from Step2): {tau}")
            print()
        except Exception as e:
            print(f"(warn) failed to read threshold.json: {e}")
            print()

    rows = _load_rows(shots_csv)
    labeled = [r for r in rows if r.label_bright in (0, 1)]

    print("=== shots.csv summary ===")
    print(f"rows_total: {len(rows)}")
    print(f"rows_labeled(label_bright present): {len(labeled)}")

    # Confusion matrix for labeled subset
    cm = {(lb, b): 0 for lb in (0, 1) for b in (0, 1)}
    for r in labeled:
        cm[(int(r.label_bright), int(r.bright))] += 1

    correct = cm[(0, 0)] + cm[(1, 1)]
    mism = cm[(0, 1)] + cm[(1, 0)]
    total = len(labeled)

    def pct(n: int) -> str:
        return "0.0%" if total == 0 else f"{(100.0 * n / total):.1f}%"

    print("\n=== confusion (label_bright -> bright) ===")
    print(f"label=0, pred=0: {cm[(0,0)]}")
    print(f"label=0, pred=1: {cm[(0,1)]}  (false positive) [{pct(cm[(0,1)])}]")
    print(f"label=1, pred=0: {cm[(1,0)]}  (false negative) [{pct(cm[(1,0)])}]")
    print(f"label=1, pred=1: {cm[(1,1)]}")
    print(f"correct: {correct} [{pct(correct)}]")
    print(f"mismatch: {mism} [{pct(mism)}]\n")

    # Detail: false positives (dark labeled but predicted bright)
    fps = [r for r in labeled if r.label_bright == 0 and r.bright == 1]
    fns = [r for r in labeled if r.label_bright == 1 and r.bright == 0]

    def group_stats(rows_: list[Row]) -> list[tuple[str, int, float | None, float | None, float | None]]:
        by = defaultdict(list)
        for rr in rows_:
            by[rr.cam_sample].append(rr)
        items = []
        for sample, rs in by.items():
            s_vals = [x.s_norm for x in rs if x.s_norm is not None]
            t_vals = [x.tau_on for x in rs if x.tau_on is not None]
            d_vals = []
            for x in rs:
                if x.s_norm is not None and x.tau_on is not None:
                    d_vals.append(x.s_norm - x.tau_on)
            items.append(
                (
                    sample or "(empty)",
                    len(rs),
                    mean(s_vals) if s_vals else None,
                    mean(t_vals) if t_vals else None,
                    mean(d_vals) if d_vals else None,
                )
            )
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def print_group(title: str, rows_: list[Row]) -> None:
        print(f"=== {title} ===")
        print(f"count: {len(rows_)}")
        if not rows_:
            print()
            return
        print("top by cam_sample:")
        for sample, n, s_mean, t_mean, d_mean in group_stats(rows_)[: max(1, args.max)]:
            print(
                f"  {n:4d}  sample={sample}"
                f"  mean(S_norm)={_fmt_float(s_mean)}"
                f"  mean(tau_on)={_fmt_float(t_mean)}"
                f"  mean(S_norm-tau_on)={_fmt_float(d_mean)}"
            )
        print()

    print_group("false positives (label=dark, pred=bright)", fps)
    print_group("false negatives (label=bright, pred=dark)", fns)

    # Quick sanity: tau consistency
    taus = [(r.tau_on, r.tau_off) for r in rows if r.tau_on is not None or r.tau_off is not None]
    if taus:
        unique = {(a, b) for (a, b) in taus}
        print("=== tau_on/off seen in shots.csv ===")
        for a, b in sorted(unique, key=lambda x: (x[0] is None, x[0] or 0.0, x[1] is None, x[1] or 0.0))[:50]:
            print(f"tau_on={_fmt_float(a)}  tau_off={_fmt_float(b)}")
        if len(unique) > 50:
            print(f"... ({len(unique)-50} more)")
        print()

    print(f"OK: analyzed {shots_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
