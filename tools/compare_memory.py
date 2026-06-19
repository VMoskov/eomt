"""
Compare memory profiles of two ViT-Adapter variants (e.g. normal vs MHA).

Usage:
    python tools/compare_memory.py --a tools/usage/.small_normal.json --b tools/usage/.small_mha.json \
        --label-a normal --label-b mha --size small --out tools/usage/small_diff.txt
"""

import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description="Compare two memory profile JSON files")
    p.add_argument("--a", required=True, help="JSON file for variant A")
    p.add_argument("--b", required=True, help="JSON file for variant B")
    p.add_argument("--label-a", default="normal", help="Label for variant A")
    p.add_argument("--label-b", default="mha", help="Label for variant B")
    p.add_argument("--size", default="", help="Model size label (small/base/large)")
    p.add_argument("--out", help="Write output to this file instead of stdout")
    return p.parse_args()


def fmt_diff(a, b, unit="GB", decimals=3):
    diff = b - a
    pct = (diff / abs(a) * 100) if a != 0 else float("inf")
    sign = "+" if diff >= 0 else "-"
    fmt = f".{decimals}f"
    return f"{sign}{abs(diff):{fmt}} {unit}  ({'+' if pct >= 0 else ''}{pct:.1f}%)"


def build_report(a, b, label_a, label_b, size):
    W = 60
    la = label_a.upper()
    lb = label_b.upper()
    lines = []

    lines.append("=" * W)
    title = f"MEMORY COMPARISON: {size}  ({label_a} vs {label_b})"
    lines.append(title.center(W))
    lines.append("=" * W)

    # --- Forward-only ---
    lines.append(f"\nForward-only (no_grad, AMP):")
    lines.append(f"  {'':30s}  {la:<12s}  {lb}")
    for i, (pa, pb) in enumerate(zip(a["fwd_peaks_gb"], b["fwd_peaks_gb"]), 1):
        lines.append(f"  BS={i} peak{'':<23s}  {pa:.3f} GB     {pb:.3f} GB")
    lines.append(f"  Per-sample avg:")
    pa_avg = a["fwd_per_sample_avg_gb"]
    pb_avg = b["fwd_per_sample_avg_gb"]
    lines.append(f"    {label_a:<10s}: {pa_avg:.3f} GB")
    lines.append(f"    {label_b:<10s}: {pb_avg:.3f} GB")
    lines.append(f"    diff      : {fmt_diff(pa_avg, pb_avg)}")

    # --- Forward + Backward ---
    if a.get("bwd_peaks_gb") and b.get("bwd_peaks_gb"):
        lines.append(f"\nForward + Backward (AMP):")
        lines.append(f"  {'':30s}  {la:<12s}  {lb}")
        for i, (pa, pb) in enumerate(zip(a["bwd_peaks_gb"], b["bwd_peaks_gb"]), 1):
            lines.append(f"  BS={i} peak{'':<23s}  {pa:.3f} GB     {pb:.3f} GB")
        lines.append(f"  Per-sample avg:")
        pa_avg = a["bwd_per_sample_avg_gb"]
        pb_avg = b["bwd_per_sample_avg_gb"]
        lines.append(f"    {label_a:<10s}: {pa_avg:.3f} GB")
        lines.append(f"    {label_b:<10s}: {pb_avg:.3f} GB")
        lines.append(f"    diff      : {fmt_diff(pa_avg, pb_avg)}")

    # --- Per-component ---
    comps_a = a.get("component_deltas_mb", {})
    comps_b = b.get("component_deltas_mb", {})
    all_keys = sorted(
        set(comps_a) | set(comps_b),
        key=lambda k: -max(abs(comps_a.get(k, 0)), abs(comps_b.get(k, 0)))
    )
    if all_keys:
        lines.append(f"\nPer-component net memory delta (BS=1, forward):")
        lines.append(f"  {'component':<30s}  {la:<14s}  {lb:<14s}  diff")
        for k in all_keys:
            va = comps_a.get(k, 0.0)
            vb = comps_b.get(k, 0.0)
            diff = vb - va
            sign_a = "+" if va >= 0 else "-"
            sign_b = "+" if vb >= 0 else "-"
            sign_d = "+" if diff >= 0 else "-"
            lines.append(
                f"  {k:<30s}  {sign_a}{abs(va):7.1f} MB     "
                f"{sign_b}{abs(vb):7.1f} MB     "
                f"{sign_d}{abs(diff):7.1f} MB"
            )

    lines.append("=" * W)
    return "\n".join(lines)


def main():
    args = parse_args()

    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)

    report = build_report(a, b, args.label_a, args.label_b, args.size)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"Comparison written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
