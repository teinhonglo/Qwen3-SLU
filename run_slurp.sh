#!/bin/bash
# dependency: torch, torchaudio, transformers, librosa, huggingface_hub

set -euo pipefail

# data config
repo_id="SLURP"
data_root="data/slurp"
exp_root="exp/slurp"
json_root="data-json/slurp"
inference_mode="--auto_latest_checkpoint"
prompt_file=""   # Optional external prompt file. Empty means using prepare_slurp_jsonl.py default prompt
attention_map_opts="" # e.g., --save_attention_map --attn_layers all --attn_mode rollout --attn_imgs_dir imgs

# training config
nj=4
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_06b.json
seed=66

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

conf_tag=$(basename -s .json $train_conf)
exp_root=$exp_root/${conf_tag}${suffix}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download/prepare SLURP and generate qwen JSONL"

    prep_cmd=(
        python local/prepare_slurp_jsonl.py
        --data-root "$data_root"
        --jsonl-root "$json_root"
    )

    if [ -n "$prompt_file" ]; then
        prep_cmd+=(--prompt-file "$prompt_file")
    fi

    "${prep_cmd[@]}"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Finetuning on SLURP"

    data_dir=$json_root
    exp_dir=$exp_root

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed $seed \
            --train_conf $train_conf \
            --train_file $data_dir/train.jsonl \
            --eval_file $data_dir/dev.jsonl \
            --output_dir $exp_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Inference on SLURP test"

    data_dir=$json_root
    exp_dir=$exp_root

    for test_set in $test_sets; do
        test_jsonl=${data_dir}/${test_set}.jsonl

        mkdir -p ${exp_dir}/${test_set}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir $exp_dir \
                --input_jsonl $test_jsonl \
                --output_root $exp_dir \
                --device cuda:0 \
                $attention_map_opts
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate SLURP predictions"

    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/slurp/evaluate_qwen.py \
            "$pred_file" "$gt_file" \
            --output ${exp_root}/${test_set}/metrics.txt
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Summary (SLURP)"

    for test_set in $test_sets; do
        metrics_file=${exp_root}/${test_set}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi

        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi
