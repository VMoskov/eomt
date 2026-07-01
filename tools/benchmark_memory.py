#!/usr/bin/env python3
"""
Training VRAM benchmark + params + GFLOPs across models.

Suite mode (recommended):
    python tools/benchmark_memory.py --suite tools/benchmark_memory_suite.yaml

Single-model mode:
    python tools/benchmark_memory.py --config configs/dinov2/coco/panoptic/eomt_small_640.yaml --name "EoMT-S"
"""

import argparse
import inspect
import importlib
import os
import sys

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Config helpers (same as benchmark_fps.py)
# ---------------------------------------------------------------------------

def import_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def instantiate(obj):
    if isinstance(obj, dict) and "class_path" in obj:
        cls = import_class(obj["class_path"])
        init_args = {k: instantiate(v) for k, v in obj.get("init_args", {}).items()}
        return cls(**init_args)
    if isinstance(obj, dict):
        return {k: instantiate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [instantiate(v) for v in obj]
    return obj


def _data_args_with_defaults(cfg: dict) -> dict:
    data_section = cfg.get("data", {})
    yaml_args = data_section.get("init_args", {})
    class_path = data_section.get("class_path", "")
    if not class_path:
        return yaml_args
    try:
        cls = import_class(class_path)
        defaults = {
            name: param.default
            for name, param in inspect.signature(cls.__init__).parameters.items()
            if name != "self" and param.default is not inspect.Parameter.empty
        }
        defaults.update(yaml_args)
        return defaults
    except Exception:
        return yaml_args


def apply_link_arguments(cfg: dict):
    data_args  = _data_args_with_defaults(cfg)
    model_args = cfg.get("model", {}).get("init_args", {})

    img_size    = data_args.get("img_size")
    num_classes = data_args.get("num_classes")
    stuff_cls   = data_args.get("stuff_classes")

    def _accepts(class_path, param):
        if not class_path:
            return False
        try:
            cls = import_class(class_path)
            return param in inspect.signature(cls.__init__).parameters
        except Exception:
            return False

    if img_size is not None:
        model_args.setdefault("img_size", img_size)
        net_section = model_args.get("network", {})
        net = net_section.get("init_args", {})
        if _accepts(net_section.get("class_path"), "img_size"):
            net.setdefault("img_size", img_size)
        enc_section = net.get("encoder", {})
        if enc_section and _accepts(enc_section.get("class_path"), "img_size"):
            enc_section.setdefault("init_args", {})
            enc_section["init_args"].setdefault("img_size", img_size)

    if num_classes is not None:
        model_args.setdefault("num_classes", num_classes)
        net_section = model_args.get("network", {})
        net = net_section.get("init_args", {})
        if _accepts(net_section.get("class_path"), "num_classes"):
            net.setdefault("num_classes", num_classes)

    if stuff_cls is not None:
        model_args.setdefault("stuff_classes", stuff_cls)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 3, 4]


def measure_peak(network, optimizer, batch_size: int, img_size, device) -> float:
    H, W = img_size

    def fn():
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(batch_size, 3, H, W, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = network(x)
        mask_logits, class_logits = outputs
        loss = mask_logits[0].sum() + class_logits[0].float().sum()
        loss.backward()
        optimizer.step()

    fn()  # warmup — also allocates AdamW moment buffers
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024**3


def count_params(network) -> tuple[int, int]:
    total     = sum(p.numel() for p in network.parameters())
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return total, trainable


def count_gflops(network, img_size, device) -> float:
    H, W = img_size
    x = torch.rand(1, 3, H, W, device=device)
    network.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(network, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    network.train()
    return total / 1e9  # GFLOPs


def run_single(name: str, config_path: str, device) -> dict:
    print(f"  Loading {name} ...")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    apply_link_arguments(cfg)

    img_size = _data_args_with_defaults(cfg).get("img_size", (640, 640))
    if isinstance(img_size, (list, tuple)):
        img_size = tuple(img_size)

    model = instantiate(cfg["model"])
    network = model.network.train().to(device)

    total_params, trainable_params = count_params(network)
    print(f"    Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    gflops = count_gflops(network, img_size, device)
    print(f"    GFLOPs: {gflops:.1f}")

    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)

    peaks = []
    for bs in BATCH_SIZES:
        gb = measure_peak(network, optimizer, bs, img_size, device)
        peaks.append(gb)
        print(f"    BS={bs}: {gb:.2f} GB")

    del optimizer, network, model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "img_size": img_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "peaks_gb": peaks,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _gb(v: float) -> str:
    return f"{v:.2f}G"


def print_table(results: list[dict]):
    W = 90
    print(f"\n{'═'*W}")
    print(f"  BENCHMARK  —  training (fwd+bwd+AdamW), AMP bfloat16, BS=1..{BATCH_SIZES[-1]}")
    print(f"{'═'*W}")

    # Params + GFLOPs table
    print(f"  {'Model':<22}  {'Params':>10}  {'Trainable':>10}  {'GFLOPs':>9}  {'img_size':>9}")
    print(f"  {'─'*(W-4)}")
    last_prefix = None
    for r in results:
        prefix = r["name"].split("-")[0] if "-" in r["name"] else r["name"]
        if last_prefix is not None and prefix != last_prefix:
            print()
        last_prefix = prefix
        h, w = r["img_size"]
        print(
            f"  {r['name']:<22}  {r['total_params']/1e6:>8.1f}M"
            f"  {r['trainable_params']/1e6:>8.1f}M"
            f"  {r['gflops']:>9.1f}"
            f"  {h}×{w}"
        )

    # VRAM table
    col = 9
    print(f"\n  {'Model':<22}" + "".join(f"{'BS='+str(bs):>{col}}" for bs in BATCH_SIZES)
          + f"  {'per-img':>8}")
    print(f"  {'─'*(W-4)}")
    last_prefix = None
    for r in results:
        prefix = r["name"].split("-")[0] if "-" in r["name"] else r["name"]
        if last_prefix is not None and prefix != last_prefix:
            print()
        last_prefix = prefix
        peaks = r["peaks_gb"]
        deltas = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        per_img = sum(deltas) / len(deltas)
        row = (f"  {r['name']:<22}"
               + "".join(f"{_gb(g):>{col}}" for g in peaks)
               + f"  {per_img:>6.2f}G")
        print(row)

    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite",  help="YAML file listing models to benchmark")
    group.add_argument("--config", help="Single model config YAML")
    parser.add_argument("--name",  default="model", help="Name for single-model mode")
    args = parser.parse_args()

    device = torch.device("cuda")

    if args.suite:
        with open(args.suite) as f:
            suite = yaml.safe_load(f)
        results = []
        for entry in suite["models"]:
            r = run_single(entry["name"], entry["config"], device)
            results.append(r)
        print_table(results)
    else:
        r = run_single(args.name, args.config, device)
        print_table([r])


if __name__ == "__main__":
    main()
