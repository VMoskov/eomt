#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

NORMAL_CFG="configs/vit_adapter_m2f/coco/panoptic"
MHA_CFG="configs/vit_adapter_m2f_mha/coco/panoptic"
OUT="usage"

for SIZE in small base large; do
    echo "=== Profiling $SIZE (normal) ==="
    python profile_memory.py \
        -c "$NORMAL_CFG/vit_adapter_m2f_${SIZE}_640.yaml" \
        --json-out "$OUT/.${SIZE}_normal.json" \
        > "$OUT/${SIZE}_normal.txt"

    echo "=== Profiling $SIZE (mha) ==="
    python profile_memory.py \
        -c "$MHA_CFG/vit_adapter_m2f_mha_${SIZE}_640.yaml" \
        --json-out "$OUT/.${SIZE}_mha.json" \
        > "$OUT/${SIZE}_mha.txt"

    echo "=== Comparing $SIZE ==="
    python compare_memory.py \
        --a "$OUT/.${SIZE}_normal.json" \
        --b "$OUT/.${SIZE}_mha.json" \
        --label-a normal \
        --label-b mha \
        --size "$SIZE" \
        --out "$OUT/${SIZE}_diff.txt"
done

echo ""
echo "Done. Files written to $OUT/:"
ls -1 "$OUT/"
