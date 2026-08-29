#!/bin/bash
set -euo pipefail

json_root="data-json/macslu"
exp_root="exp/macslu"
train_conf="conf/macslu_qwen3_asr_06b.json"
test_sets="test"
gpuid=0
inference_mode="--auto_latest_checkpoint"
alpha=1.0
contrast_scope="both"
max_new_tokens=256
silence_value=0.0
plausibility_alpha=0.0

. ./local/parse_options.sh
. ./path.sh

conf_tag=$(basename -s .json "$train_conf")
exp_dir="${exp_root}/${conf_tag}"

for test_set in $test_sets; do
    output_dir="${exp_dir}/${test_set}_silence_contrast"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_test_silence_contrast.py \
            $inference_mode \
            --exp_dir "$exp_dir" \
            --input_jsonl "${json_root}/${test_set}.jsonl" \
            --output_dir "$output_dir" \
            --device cuda:0 \
            --max_new_tokens "$max_new_tokens" \
            --alpha "$alpha" \
            --contrast_scope "$contrast_scope" \
            --silence_value "$silence_value" \
            --plausibility_alpha "$plausibility_alpha"

    for mode in silence first_step all_steps; do
        pred_file="${output_dir}/predictions_${mode}.jsonl"
        [ -f "$pred_file" ] || continue
        python local/metrics.py \
            --output_dir "${output_dir}/${mode}" \
            "$pred_file" "${json_root}/${test_set}.jsonl" \
            | tee "${output_dir}/metrics_${mode}.txt"
    done
done
