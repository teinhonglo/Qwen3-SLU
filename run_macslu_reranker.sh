#!/bin/bash
# Clean-test Qwen3-Reranker control.
# This script generates clean-test N-best hypotheses, reranks them as text,
# and evaluates the selected candidate. It never creates noisy data.
set -euo pipefail

json_root="data-json/macslu_fixed"
exp_root="exp/macslu_reranker"
src_exp_dir="exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead"
inference_mode="--auto_latest_checkpoint"
nbest_decoding_conf="conf/decoding/nbest_decoding.json"
reranker_model="Qwen/Qwen3-Reranker-0.6B"
test_set="test"
gpuid=0
stage=0
stop_stage=3

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$nbest_decoding_conf" ]; then
    echo "[ERROR] N-best decoding config not found: $nbest_decoding_conf"
    exit 1
fi
if [ ! -f "\${json_root}/\${test_set}.jsonl" ]; then
    echo "[ERROR] clean test JSONL not found: \${json_root}/\${test_set}.jsonl"
    exit 1
fi
if [ ! -d "$src_exp_dir" ]; then
    echo "[ERROR] source ASR experiment not found: $src_exp_dir"
    exit 1
fi

nbest_conf_name=$(basename -s .json "$nbest_decoding_conf")
run_dir="\${exp_root}/\${test_set}_\${nbest_conf_name}"
nbest_dir="\${run_dir}/nbest"
nbest_file="\${nbest_dir}/\${test_set}.jsonl"
prediction_file="\${run_dir}/predictions.jsonl"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Validate clean-test reranker inputs"
    mkdir -p "$nbest_dir"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate clean-test N-best hypotheses (N=10)"
    if [ -s "$nbest_file" ]; then
        echo "[SKIP] Existing clean-test N-best file: $nbest_file"
    else
        CUDA_VISIBLE_DEVICES="$gpuid" python finetuning/qwen3_asr_test.py \
            $inference_mode \
            --exp_dir "$src_exp_dir" \
            --input_jsonl "\${json_root}/\${test_set}.jsonl" \
            --output_root "$run_dir" \
            --device cuda:0 \
            --decoding_conf "$nbest_decoding_conf" \
            --output_nbest_jsonl_dir "$nbest_dir"
    fi
    if [ ! -s "$nbest_file" ]; then
        echo "[ERROR] clean-test N-best output not found: $nbest_file"
        exit 1
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Rerank clean-test N-best with $reranker_model"
    if [ ! -s "$nbest_file" ]; then
        echo "[ERROR] N-best input not found: $nbest_file"
        exit 1
    fi
    if [ -s "$prediction_file" ]; then
        echo "[SKIP] Existing reranker predictions: $prediction_file"
    else
        CUDA_VISIBLE_DEVICES="$gpuid" python local/rerank_nbest_qwen3.py \
            --input_jsonl "$nbest_file" \
            --output_jsonl "$prediction_file" \
            --model_name "$reranker_model" \
            --device cuda:0 \
            --n_best 10
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate clean-test reranker predictions"
    if [ ! -s "$prediction_file" ]; then
        echo "[ERROR] reranker prediction file not found: $prediction_file"
        exit 1
    fi
    python local/metrics.py \
        --output_dir "$run_dir" \
        "$prediction_file" \
        "\${json_root}/\${test_set}.jsonl" \
        | tee "\${run_dir}/metrics.txt"
fi
