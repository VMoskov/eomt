"""
GPU memory profiler for ViT-Adapter M2F models.

Measures peak memory at BS=1, BS=2, BS=3 to estimate per-sample cost,
and reports net memory delta per model component during the forward pass.

Usage:
    python tools/profile_memory.py -c configs/vit_adapter_m2f/coco/panoptic/vit_adapter_m2f_small_640.yaml
    python tools/profile_memory.py -c <config> --json-out tools/usage/.small_normal.json
"""

import argparse
import json
import warnings

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="GPU memory profiler for ViTAdapterM2F")
    p.add_argument("-c", "--config", required=True,
                   help="Path to config YAML")
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device index (default: 0)")
    p.add_argument("--no_backward", action="store_true",
                   help="Skip forward+backward measurement")
    p.add_argument("--json-out", metavar="PATH",
                   help="Also save results as JSON (for compare_memory.py)")
    return p.parse_args()


def load_network(config_path: str):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module`.*"
    )

    from jsonargparse import ArgumentParser as JParser
    from training.lightning_module import LightningModule

    parser = JParser()
    parser.add_subclass_arguments(LightningModule, "model")
    cfg = parser.parse_object({"model": raw["model"]})
    cfg_init = parser.instantiate_classes(cfg)
    lightning_module = cfg_init.model

    img_size = raw["model"]["init_args"]["img_size"]
    return lightning_module.network, img_size


def measure_peak(fn, device):
    """Warmup once, then reset peak stats and measure."""
    fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def make_forward_fn(network, batch_size, img_size, device, with_backward):
    H, W = img_size

    def fn():
        x = torch.rand(batch_size, 3, H, W, device=device, dtype=torch.float32)
        if with_backward:
            with torch.autocast("cuda", dtype=torch.float16):
                mask_logits, class_logits = network(x)
            loss = mask_logits[0].sum() + class_logits[0].float().sum()
            loss.backward()
            network.zero_grad(set_to_none=True)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                network(x)

    return fn


def profile_components(network, img_size, device):
    """Register pre/post hooks on key components, run one forward, return deltas."""
    H, W = img_size
    targets = {
        "encoder":              network.encoder,
        "encoder.backbone":     network.encoder.backbone,
        "encoder.spm":          network.encoder.spm,
        "encoder.interactions": network.encoder.interactions,
        "pixel_decoder":        network.pixel_decoder,
        "transformer_module":   network.transformer_module,
    }

    pre_mem = {}
    results = {}
    handles = []

    for name, module in targets.items():
        def make_pre(n):
            def hook(mod, inp):
                torch.cuda.synchronize()
                pre_mem[n] = torch.cuda.memory_allocated()
            return hook

        def make_post(n):
            def hook(mod, inp, out):
                torch.cuda.synchronize()
                results[n] = torch.cuda.memory_allocated() - pre_mem.get(n, 0)
            return hook

        handles.append(module.register_forward_pre_hook(make_pre(name)))
        handles.append(module.register_forward_hook(make_post(name)))

    x = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        network(x)

    for h in handles:
        h.remove()

    return results


def print_report(fwd_peaks, bwd_peaks, component_deltas):
    GB = 1e9
    MB = 1e6

    def per_sample_stats(peaks):
        deltas = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        avg = sum(deltas) / len(deltas)
        return deltas, avg

    print("\n" + "=" * 60)
    print("MEMORY PROFILE REPORT")
    print("=" * 60)

    print(f"\nForward-only (no_grad, AMP):")
    for i, p in enumerate(fwd_peaks, 1):
        print(f"  BS={i} peak  : {p / GB:.3f} GB")
    deltas, avg = per_sample_stats(fwd_peaks)
    delta_strs = ", ".join(f"{d / GB:.3f}" for d in deltas)
    print(f"  Per-sample deltas : [{delta_strs}] GB  →  avg {avg / GB:.3f} GB")

    if bwd_peaks:
        print(f"\nForward + Backward (AMP):")
        for i, p in enumerate(bwd_peaks, 1):
            print(f"  BS={i} peak  : {p / GB:.3f} GB")
        deltas, avg = per_sample_stats(bwd_peaks)
        delta_strs = ", ".join(f"{d / GB:.3f}" for d in deltas)
        print(f"  Per-sample deltas : [{delta_strs}] GB  →  avg {avg / GB:.3f} GB")

    if component_deltas:
        print(f"\nPer-component net memory delta (BS=1, forward):")
        print(f"  (positive = net allocation, negative = net free)")
        print(f"  Note: pixel_decoder + transformer_module run in fp32 (AMP disabled internally)")
        sorted_items = sorted(component_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for name, delta in sorted_items:
            sign = "+" if delta >= 0 else "-"
            print(f"  {name:<30s}: {sign}{abs(delta) / MB:7.1f} MB")

    print("=" * 60)


def main():
    args = parse_args()
    device = args.device
    torch.cuda.set_device(device)

    print(f"Loading model from: {args.config}")
    network, img_size = load_network(args.config)
    network = network.eval().to(device)

    print(f"Image size: {img_size}, Device: cuda:{device}")

    print("\n--- Forward-only ---")
    fwd_peaks = []
    for bs in [1, 2, 3, 4]:
        peak = measure_peak(make_forward_fn(network, bs, img_size, device, False), device)
        fwd_peaks.append(peak)
        print(f"  BS={bs}: {peak / 1e9:.3f} GB")

    bwd_peaks = []
    if not args.no_backward:
        network.train()
        print("\n--- Forward + Backward ---")
        for bs in [1, 2, 3, 4]:
            peak = measure_peak(make_forward_fn(network, bs, img_size, device, True), device)
            bwd_peaks.append(peak)
            print(f"  BS={bs}: {peak / 1e9:.3f} GB")
        network.eval()

    print("\n--- Per-component hooks (BS=1, forward) ---")
    component_deltas = profile_components(network, img_size, device)

    print_report(fwd_peaks, bwd_peaks, component_deltas)

    if args.json_out:
        def avg_delta(peaks):
            deltas = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
            return sum(deltas) / len(deltas)

        data = {
            "config": args.config,
            "img_size": img_size,
            "fwd_peaks_gb": [p / 1e9 for p in fwd_peaks],
            "fwd_per_sample_avg_gb": avg_delta(fwd_peaks) / 1e9,
            "bwd_peaks_gb": [p / 1e9 for p in bwd_peaks],
            "bwd_per_sample_avg_gb": avg_delta(bwd_peaks) / 1e9 if bwd_peaks else None,
            "component_deltas_mb": {k: v / 1e6 for k, v in component_deltas.items()},
        }
        with open(args.json_out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
