#!/bin/bash
# Evaluate whether a trained StructSFT checkpoint actually learned PII and CDI.

set -euo pipefail

# data / experiment config
src_json_root=data-json/macslu_fixed
eval_json_root=data-json/macslu_fixed_structprompt_eval
exp_root="exp/macslu_fixed_structprompt"
train_conf=conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead.json
decoding_conf=conf/decoding/basic_decoding.json
inference_mode="--auto_latest_checkpoint" # or --auto_best_checkpoint
split=dev

# evaluation config
gpuid=0
seed=66
suffix=
cdi_pairs_per_class=1
cdi_max_anchor_intent_count=3

# stage
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi
if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding_conf not found: $decoding_conf"
    exit 1
fi
if [ ! -f "${src_json_root}/${split}.jsonl" ]; then
    echo "[ERROR] source split not found: ${src_json_root}/${split}.jsonl"
    exit 1
fi

conf_tag=$(basename -s .json "$train_conf")
decoding_conf_name=$(basename -s .json "$decoding_conf")
exp_root=${exp_root}/${conf_tag}${suffix}
aux_output_root=${exp_root}/aux_eval

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "Stage 0: Prepare held-out PII/CDI evaluation JSONL"
    python local/prepare_macslu_structprompt_eval_jsonl.py \
        --src-json-root "$src_json_root" \
        --output-root "$eval_json_root" \
        --split "$split" \
        --seed "$seed" \
        --cdi-pairs-per-class "$cdi_pairs_per_class" \
        --cdi-max-anchor-intent-count "$cdi_max_anchor_intent_count"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: PII inference"
    mkdir -p "$aux_output_root"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_test_aux.py \
            $inference_mode \
            --exp_dir "$exp_root" \
            --input_jsonl "${eval_json_root}/${split}_pii.jsonl" \
            --output_root "$aux_output_root" \
            --device cuda:0 \
            --decoding_conf "$decoding_conf" \
            --seed "$seed"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: CDI inference"
    mkdir -p "$aux_output_root"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_test_aux.py \
            $inference_mode \
            --exp_dir "$exp_root" \
            --input_jsonl "${eval_json_root}/${split}_cdi.jsonl" \
            --output_root "$aux_output_root" \
            --device cuda:0 \
            --decoding_conf "$decoding_conf" \
            --seed "$seed"
fi

pii_dir=${aux_output_root}/${split}_pii_${decoding_conf_name}
cdi_dir=${aux_output_root}/${split}_cdi_${decoding_conf_name}

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Stage 3: Compute PII/CDI metrics"

    if [ ! -f "${pii_dir}/predictions.jsonl" ]; then
        echo "[ERROR] PII predictions not found: ${pii_dir}/predictions.jsonl"
        exit 1
    fi
    if [ ! -f "${cdi_dir}/predictions.jsonl" ]; then
        echo "[ERROR] CDI predictions not found: ${cdi_dir}/predictions.jsonl"
        exit 1
    fi

    python local/metrics_macslu_structprompt_aux.py \
        --task pii \
        --pred-file "${pii_dir}/predictions.jsonl" \
        --gt-file "${eval_json_root}/${split}_pii.jsonl" \
        --output-dir "$pii_dir"

    python local/metrics_macslu_structprompt_aux.py \
        --task cdi \
        --pred-file "${cdi_dir}/predictions.jsonl" \
        --gt-file "${eval_json_root}/${split}_cdi.jsonl" \
        --output-dir "$cdi_dir"
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Stage 4: Summary"
    echo "========== PII =========="
    cat "${pii_dir}/metrics.txt"
    echo
    echo "========== CDI =========="
    cat "${cdi_dir}/metrics.txt"
    echo
    echo "Evaluation data summary: ${eval_json_root}/${split}_summary.json"
fi
