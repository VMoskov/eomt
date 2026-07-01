#!/usr/bin/env python3
"""
Inference FPS benchmark using the actual validation dataloader.
Separately measures: data loading, preprocessing, encoder, decoder, postprocessing.

Single model:
    python tools/benchmark_fps.py \
        --config configs/vit_adapter_m2f/ade20k/semantic/vit_adapter_m2f_large_512.yaml \
        --ckpt /lustre/home/vmoskov/checkpoints/ade20k_vit_adapter_m2f_large_512/best-epoch=60.ckpt \
        --data_path /lustre/home/vmoskov/datasets/ade20k/semantic \
        [--windowed_inference true|false]

Suite (all models at once):
    python tools/benchmark_fps.py --suite tools/benchmark_suite.yaml
"""

import argparse
import copy
import importlib
import inspect
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Config helpers
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
    """Return data init_args merged with the datamodule class's __init__ defaults.
    Needed because EoMT configs omit img_size/num_classes from the YAML entirely,
    relying on LightningCLI to read them from the class signature."""
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
        defaults.update(yaml_args)  # YAML always wins over class defaults
        return defaults
    except Exception:
        return yaml_args


def apply_link_arguments(cfg: dict):
    """Replicates main.py link_arguments: push img_size/num_classes/stuff_classes
    from data config into model config where missing."""
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


def load_model(model_cfg: dict, ckpt_path: str, windowed_override=None):
    model_cfg = copy.deepcopy(model_cfg)
    if windowed_override is not None and "windowed_inference" in model_cfg.get("init_args", {}):
        model_cfg["init_args"]["windowed_inference"] = windowed_override
    model = instantiate(model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def preprocess_semantic(model, imgs):
    if model.windowed_inference:
        crops, origins = model.window_imgs_semantic(imgs)
        return crops, {"origins": origins, "img_sizes": [img.shape[-2:] for img in imgs]}
    else:
        transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
        return transformed, {"img_sizes": [img.shape[-2:] for img in imgs]}


def preprocess_panoptic(model, imgs):
    transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
    return transformed, {"img_sizes": [img.shape[-2:] for img in imgs]}


def postprocess_semantic(model, outputs, meta):
    mask_logits_per_layer, class_logits_per_layer = outputs
    for mask_logits, class_logits in zip(mask_logits_per_layer, class_logits_per_layer):
        mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
        if model.windowed_inference:
            crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            model.revert_window_logits_semantic(crop_logits, meta["origins"], meta["img_sizes"])
        else:
            pixel_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            model.revert_resize_and_pad_logits_instance_panoptic(pixel_logits, meta["img_sizes"])


def postprocess_panoptic(model, outputs, meta):
    mask_logits_per_layer, class_logits_per_layer = outputs
    for mask_logits, _ in zip(mask_logits_per_layer, class_logits_per_layer):
        mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
        model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, meta["img_sizes"])


def get_fns(model):
    if "Semantic" in type(model).__name__:
        return preprocess_semantic, postprocess_semantic
    return preprocess_panoptic, postprocess_panoptic


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def sync():
    torch.cuda.synchronize()


def benchmark(model, val_loader, preprocess_fn, postprocess_fn, warmup_images, num_images, device):
    model = model.to(device).eval()

    print(f"  Warming up ({warmup_images} images)...")
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_loader):
            if i >= warmup_images:
                break
            imgs = tuple(img.to(device) for img in imgs)
            x, meta = preprocess_fn(model, imgs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            postprocess_fn(model, out, meta)
            sync()

    limit = num_images if num_images is not None else float("inf")
    desc = "full val set" if num_images is None else f"{num_images} images"
    print(f"  Measuring ({desc})...")

    t_load = t_pre = t_fwd = t_enc = t_dec = t_post = 0.0
    enc_measured = [False]
    total = 0
    data_iter = iter(val_loader)

    with torch.no_grad():
        while total < limit:
            # ── Data loading ──────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                imgs, _ = next(data_iter)
            except StopIteration:
                break
            imgs = tuple(img.to(device) for img in imgs)
            sync()
            t_load += time.perf_counter() - t0

            # ── Preprocessing ─────────────────────────────────────────────
            sync()
            t0 = time.perf_counter()
            x, meta = preprocess_fn(model, imgs)
            sync()
            t_pre += time.perf_counter() - t0

            # ── Forward pass (with encoder/decoder split via hooks) ────────
            enc_times = []

            def _pre(_m, _i):
                sync()
                enc_times.append(time.perf_counter())

            def _post(_m, _i, _o):
                sync()
                enc_times.append(time.perf_counter())

            h_pre  = model.network.encoder.register_forward_pre_hook(_pre)
            h_post = model.network.encoder.register_forward_hook(_post)

            sync()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            sync()
            dt_fwd = time.perf_counter() - t0
            t_fwd += dt_fwd

            h_pre.remove()
            h_post.remove()

            # enc_times has 2 entries only if encoder was called as a unit
            if len(enc_times) == 2:
                dt_enc = enc_times[1] - enc_times[0]
                t_enc += dt_enc
                t_dec += dt_fwd - dt_enc
                enc_measured[0] = True

            # ── Postprocessing ────────────────────────────────────────────
            sync()
            t0 = time.perf_counter()
            postprocess_fn(model, out, meta)
            sync()
            t_post += time.perf_counter() - t0

            total += 1

    mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    return {
        "n": total,
        "t_load": t_load,
        "t_pre": t_pre,
        "t_fwd": t_fwd,
        "t_enc": t_enc,
        "t_dec": t_dec,
        "t_post": t_post,
        "enc_measured": enc_measured[0],
        "mem_gb": mem_gb,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_result(name: str, r: dict):
    n = r["n"]
    t_load, t_pre, t_fwd, t_enc, t_dec, t_post = (
        r["t_load"], r["t_pre"], r["t_fwd"], r["t_enc"], r["t_dec"], r["t_post"]
    )
    t_total = t_load + t_pre + t_fwd + t_post

    def ms(t): return t / n * 1000
    def fps(t): return n / t
    def pct(t): return 100 * t / t_total

    W = 57
    print(f"\n{'─'*W}")
    print(f"  {name}")
    print(f"  Images: {n}  |  Peak VRAM: {r['mem_gb']:.2f} GB")
    print(f"  {'Stage':<22}  {'ms/img':>8}  {'FPS':>7}  {'%':>5}")
    print(f"  {'─'*(W-4)}")
    print(f"  {'Data loading':<22}  {ms(t_load):>8.1f}  {fps(t_load):>7.1f}  {pct(t_load):>4.1f}%")
    print(f"  {'Preprocessing':<22}  {ms(t_pre):>8.1f}  {fps(t_pre):>7.1f}  {pct(t_pre):>4.1f}%")
    if r["enc_measured"]:
        print(f"  {'  └ Encoder':<22}  {ms(t_enc):>8.1f}  {fps(t_enc):>7.1f}  {pct(t_enc):>4.1f}%")
        print(f"  {'  └ Decoder':<22}  {ms(t_dec):>8.1f}  {fps(t_dec):>7.1f}  {pct(t_dec):>4.1f}%")
    print(f"  {'Forward pass':<22}  {ms(t_fwd):>8.1f}  {fps(t_fwd):>7.1f}  {pct(t_fwd):>4.1f}%")
    print(f"  {'Postprocessing':<22}  {ms(t_post):>8.1f}  {fps(t_post):>7.1f}  {pct(t_post):>4.1f}%")
    print(f"  {'─'*(W-4)}")
    print(f"  {'Total':<22}  {ms(t_total):>8.1f}  {fps(t_total):>7.1f}  100.0%")
    if not r["enc_measured"]:
        print(f"  (enc/dec split: N/A — integrated architecture)")
    print(f"{'─'*W}")


def print_summary(results: list[tuple[str, dict]]):
    if len(results) < 2:
        return

    print(f"\n\n{'═'*82}")
    print(f"  SUMMARY")
    print(f"{'═'*82}")

    def ms(r, key): return r[key] / r["n"] * 1000
    def na_or(r, key): return f"{ms(r, key):>6.1f}" if r["enc_measured"] else "   N/A"

    hdr = f"  {'Model':<30}  {'Enc':>6}  {'Dec':>6}  {'Fwd':>6}  {'Total':>7}  {'FPS':>6}  {'VRAM':>6}"
    print(hdr)
    print(f"  {'─'*76}")
    for name, r in results:
        t_total = r["t_load"] + r["t_pre"] + r["t_fwd"] + r["t_post"]
        fps = r["n"] / t_total
        total_ms = t_total / r["n"] * 1000
        enc_s = na_or(r, "t_enc")
        dec_s = na_or(r, "t_dec")
        fwd_s = f"{ms(r, 't_fwd'):>6.1f}"
        label = name[:30]
        print(f"  {label:<30}  {enc_s}  {dec_s}  {fwd_s}  {total_ms:>7.1f}  {fps:>6.1f}  {r['mem_gb']:>5.2f}G")
    print(f"{'═'*82}\n")


# ---------------------------------------------------------------------------
# Single-model runner (shared by both modes)
# ---------------------------------------------------------------------------

def run_single(entry: dict, warmup_images: int, num_images, device):
    """Load, benchmark, and unload one model. Returns (name, result_dict)."""
    name       = entry["name"]
    config     = entry["config"]
    ckpt       = entry["ckpt"]
    data_path  = entry["data_path"]
    windowed   = entry.get("windowed_inference", None)

    print(f"\n{'━'*57}")
    print(f"  {name}")
    print(f"{'━'*57}")
    print(f"  Config : {config}")
    print(f"  Ckpt   : {ckpt}")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    apply_link_arguments(cfg)

    model = load_model(cfg["model"], ckpt, windowed)

    data_cfg = copy.deepcopy(cfg["data"])
    data_cfg.setdefault("init_args", {})
    data_cfg["init_args"]["path"] = data_path
    data_cfg["init_args"]["batch_size"] = 1
    datamodule = instantiate(data_cfg)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()

    preprocess_fn, postprocess_fn = get_fns(model)

    mode = f"windowed={model.windowed_inference}" if hasattr(model, "windowed_inference") else "panoptic"
    print(f"  Mode   : {mode}\n")

    torch.cuda.reset_peak_memory_stats(device)
    result = benchmark(model, val_loader, preprocess_fn, postprocess_fn,
                       warmup_images, num_images, device)

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return name, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite",  help="YAML file listing multiple models to benchmark")
    group.add_argument("--config", help="Single model config YAML")

    parser.add_argument("--ckpt",      help="Checkpoint path (single-model mode)")
    parser.add_argument("--data_path", help="Dataset path (single-model mode)")
    parser.add_argument("--warmup_images", type=int, default=100)
    parser.add_argument("--num_images",    type=int, default=None,
                        help="Images to benchmark per model (default: full val set)")
    parser.add_argument("--windowed_inference", type=lambda x: x.lower() == "true",
                        default=None, help="Override windowed_inference (true/false)")
    args = parser.parse_args()

    device = torch.device("cuda")

    if args.suite:
        with open(args.suite) as f:
            suite = yaml.safe_load(f)
        entries = suite["models"]
        all_results = []
        for entry in entries:
            name, result = run_single(entry, args.warmup_images, args.num_images, device)
            print_result(name, result)
            all_results.append((name, result))
        print_summary(all_results)

    else:
        if not args.ckpt or not args.data_path:
            parser.error("--ckpt and --data_path are required in single-model mode")
        entry = {
            "name": os.path.basename(args.config).replace(".yaml", ""),
            "config": args.config,
            "ckpt": args.ckpt,
            "data_path": args.data_path,
            "windowed_inference": args.windowed_inference,
        }
        name, result = run_single(entry, args.warmup_images, args.num_images, device)
        print_result(name, result)


if __name__ == "__main__":
    main()
