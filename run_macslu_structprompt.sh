#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
src_json_root=data-json/macslu_fixed
json_root=${src_json_root}_structprompt
exp_root="exp/macslu_structprompt"
labels_file="data/macslu/labels.txt"
label_mapping_file="data/macslu/labels_zh_en.txt"
inference_mode="--auto_latest_checkpoint"
attention_map_opts="" # e.g., --save_attention_map --attn_layers all --attn_mode rollout --attn_imgs_dir imgs
decoding_conf="conf/decoding/basic_decoding.json"
slu_repeat=2

# training config
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_06b.json
seed=66
checkpoint=

# stage
stage=0
stop_stage=1000
test_sets="test"

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

conf_tag=$(basename -s .json "$train_conf")
decoding_conf_name=$(basename -s .json "$decoding_conf")
exp_root=${exp_root}/${conf_tag}${suffix}

if [ -n "$checkpoint" ]; then
    training_opts=(--resume_from "$checkpoint" --resume 1)
else
    training_opts=()
fi

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "Stage 0: Prepare structure-aware multi-prompt MAC-SLU jsonl"
    for split in train dev test; do
        if [ ! -f "${src_json_root}/${split}.jsonl" ]; then
            echo "[ERROR] Required source jsonl not found: ${src_json_root}/${split}.jsonl"
            echo "[ERROR] Prepare fixed data first, e.g.: ./run_macslu_fixed.sh --stage 1 --stop_stage 2"
            exit 1
        fi
    done
    python local/prepare_macslu_structprompt_jsonl.py \
        --src-json-root "$src_json_root" \
        --json-root "$json_root" \
        --splits train dev test \
        --expand-splits train \
        --slu-repeat "$slu_repeat"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: Finetuning on structure-aware multi-prompt MAC-SLU"
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed "$seed" "${training_opts[@]}" \
            --train_conf "$train_conf" \
            --train_file "${json_root}/train.jsonl" \
            --eval_file "${json_root}/dev.jsonl" \
            --output_dir "$exp_root"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: Inference on full-SLU MAC-SLU test"
    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl
        mkdir -p "${exp_root}/${test_set}_${decoding_conf_name}"
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode --exp_dir "$exp_root" --input_jsonl "$test_jsonl" \
                --output_root "$exp_root" --device cuda:0 \
                --decoding_conf "$decoding_conf" $attention_map_opts
    done
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Stage 3: Evaluate MAC-SLU predictions"
    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}_${decoding_conf_name}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi
        python local/metrics.py --output_dir "${exp_root}/${test_set}_${decoding_conf_name}" "$pred_file" "$gt_file" \
            | tee "${exp_root}/${test_set}_${decoding_conf_name}/metrics.txt"
    done
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Stage 4: Plot MAC-SLU evaluation charts"
    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}_${decoding_conf_name}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        output_dir=${exp_root}/${test_set}_${decoding_conf_name}
        if [ ! -f "$pred_file" ] || [ ! -f "$gt_file" ]; then
            echo "[WARNING] prediction or ground-truth file not found for $test_set"
            continue
        fi
        python local/plot_macslu_evaluation.py \
            --pred_file "$pred_file" --gt_file "$gt_file" \
            --train_file "${json_root}/train.jsonl" \
            --labels_file "$labels_file" --label_mapping_file "$label_mapping_file" \
            --output_dir "$output_dir"
    done
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Stage 5: Summary (structure-aware multi-prompt MAC-SLU)"
    for test_set in $test_sets; do
        metrics_file=${exp_root}/${test_set}_${decoding_conf_name}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi
        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi
