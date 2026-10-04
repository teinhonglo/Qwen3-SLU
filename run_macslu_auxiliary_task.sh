#!/bin/bash
# Reproduce the Figure 3 target-serialization ablation by preparing each
# auxiliary-task variant and delegating training, inference, and evaluation to
# the existing run_macslu.sh entrypoint.

set -euo pipefail

help_message="Usage: $0 [options]

Stage 0 prepares the selected Figure 3 JSONL variants. Stages 1--5 are passed
through to run_macslu.sh for every selected variant.

Each variant uses matching data and experiment roots:
  data-json/macslu_fixed_auxtask_<variant>
  exp/macslu_fixed_auxtask_<variant>
run_macslu.sh then appends its config tag under the experiment root.

Example:
  $0 --stage 0 --stop_stage 3 --gpuid 0
  $0 --stage 1 --stop_stage 3 --variants \"vanilla asr\" --gpuid 0"

# data and experiment config
src_json_root="data-json/macslu_fixed"
json_root_prefix="data-json/macslu_fixed_auxtask"
exp_root_prefix="exp/macslu_fixed_auxtask"

# Variant-list format:
# - Spaces separate independent experiments (e.g., "asr asr_nsems" runs two).
# - Underscores combine targets within one variant (e.g., "asr_nsems" is one).
# Each variant is appended to both prefixes above. For example, "asr_nsems"
# uses data-json/macslu_fixed_auxtask_asr_nsems as json_root and passes
# exp/macslu_fixed_auxtask_asr_nsems as exp_root to run_macslu.sh.
variants="vanilla asr nsems asr_nsems nslots asr_nslots asr_nsems_nslots"
decoding_conf="conf/decoding/basic_decoding.json"
inference_mode="--auto_latest_checkpoint"

# training config
gpuid=0
train_conf="conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead.json"
seed=66

# stage 0: prepare data; stages 1--5: run_macslu.sh stages
stage=0
stop_stage=3
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

supported_variants="vanilla asr nsems asr_nsems nslots asr_nslots asr_nsems_nslots"
for variant in $variants; do
    if [[ " $supported_variants " != *" $variant "* ]]; then
        echo "[ERROR] Unsupported variant: $variant"
        echo "[ERROR] Supported variants: $supported_variants"
        exit 1
    fi
done

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "Stage 0: Prepare MAC-SLU auxiliary-task variants for Figure 3"

    python local/prepare_macslu_auxiliary_task_jsonl.py \
        --src-json-root "$src_json_root" \
        --json-root-prefix "$json_root_prefix" \
        --splits train dev test \
        --variants $variants
fi

run_stage=$stage
if [ "$run_stage" -lt 1 ]; then
    run_stage=1
fi

if [ "$stop_stage" -ge 1 ]; then
    for variant in $variants; do
        variant_json_root="${json_root_prefix}_${variant}"
        variant_exp_root="${exp_root_prefix}_${variant}"
        for split in train dev test; do
            if [ ! -f "${variant_json_root}/${split}.jsonl" ]; then
                echo "[ERROR] Required JSONL not found: ${variant_json_root}/${split}.jsonl"
                echo "[ERROR] Run stage 0 first or include it in the current stage range."
                exit 1
            fi
        done

        echo "Running run_macslu.sh for auxiliary-task variant: $variant"
        ./run_macslu.sh \
            --stage "$run_stage" \
            --stop_stage "$stop_stage" \
            --json_root "$variant_json_root" \
            --exp_root "$variant_exp_root" \
            --train_conf "$train_conf" \
            --decoding_conf "$decoding_conf" \
            --inference_mode "$inference_mode" \
            --test_sets "$test_sets" \
            --gpuid "$gpuid" \
            --seed "$seed"
    done
fi
