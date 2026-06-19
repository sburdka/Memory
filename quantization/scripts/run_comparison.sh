#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Full baseline vs MiLo comparison pipeline.
# Runs on whichever Azure GPU VM you're on — just pass the config file.
#
# Usage:
#   bash quantization/scripts/run_comparison.sh \
#       --config quantization/configs/deepseek_a100.yaml \
#       --results-dir results/deepseek_a100
#
#   bash quantization/scripts/run_comparison.sh \
#       --config quantization/configs/deepseek_h100.yaml \
#       --results-dir results/deepseek_h100
#
# Optional flags:
#   --tasks "arc_easy arc_challenge hellaswag winogrande"   # zero-shot tasks
#   --skip-compress                                          # skip if already done
#   --skip-ppl                                              # skip perplexity (faster)
# -----------------------------------------------------------------------
set -euo pipefail

CONFIG=""
RESULTS_DIR="results"
TASKS=""
SKIP_COMPRESS=false
SKIP_PPL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";       shift 2 ;;
        --results-dir)   RESULTS_DIR="$2";  shift 2 ;;
        --tasks)         TASKS="$2";        shift 2 ;;
        --skip-compress) SKIP_COMPRESS=true; shift ;;
        --skip-ppl)      SKIP_PPL=true;     shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "Error: --config is required"; exit 1; }

# Parse fields from YAML config
MODEL_ID=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['model_id'])")
ARCH=$(python    -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['arch'])")
DTYPE=$(python   -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model'].get('torch_dtype','bfloat16'))")
COMPRESSED_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output_dir'])")

mkdir -p "$RESULTS_DIR"
BASELINE_JSON="$RESULTS_DIR/baseline.json"
COMPRESSED_JSON="$RESULTS_DIR/compressed.json"

EVAL_ARGS="--model-id $MODEL_ID --arch $ARCH --dtype $DTYPE"
[[ -n "$TASKS" ]] && EVAL_ARGS="$EVAL_ARGS --tasks $TASKS"
[[ "$SKIP_PPL" == true ]] && EVAL_ARGS="$EVAL_ARGS --skip-ppl"

echo "========================================================"
echo "  Config       : $CONFIG"
echo "  Model        : $MODEL_ID  ($ARCH)"
echo "  Results dir  : $RESULTS_DIR"
echo "  Compressed → : $COMPRESSED_DIR"
echo "========================================================"

# ------------------------------------------------------------------
# STEP 1: Compress (skip if already done or --skip-compress passed)
# ------------------------------------------------------------------
if [[ "$SKIP_COMPRESS" == false ]]; then
    echo ""
    echo "[ STEP 1 / 3 ]  Compressing with MiLo ..."
    python quantization/scripts/run_deepseek.py --config "$CONFIG"
else
    echo "[ STEP 1 / 3 ]  Compression skipped (--skip-compress)"
fi

# ------------------------------------------------------------------
# STEP 2: Evaluate baseline  (full fp16/bf16 model)
# ------------------------------------------------------------------
echo ""
echo "[ STEP 2 / 3 ]  Evaluating BASELINE ..."
python quantization/scripts/evaluate_model.py \
    --mode baseline \
    $EVAL_ARGS \
    --output "$BASELINE_JSON"

# ------------------------------------------------------------------
# STEP 3: Evaluate compressed
# ------------------------------------------------------------------
echo ""
echo "[ STEP 3 / 3 ]  Evaluating COMPRESSED ..."
python quantization/scripts/evaluate_model.py \
    --mode compressed \
    $EVAL_ARGS \
    --compressed-dir "$COMPRESSED_DIR" \
    --output "$COMPRESSED_JSON"

# ------------------------------------------------------------------
# FINAL: Print comparison table
# ------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  COMPARISON TABLE"
echo "========================================================"
python quantization/scripts/compare_results.py \
    --baseline "$BASELINE_JSON" \
    --compressed "$COMPRESSED_JSON"

echo "Raw results saved to: $RESULTS_DIR/"
