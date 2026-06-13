#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
data_root="data/macslu"
json_root="data-json/macslu_fixed"
curriculum_json_root="data-json/macslu_fixed_curriculum"
exp_root="exp/macslu_fixed"
inference_mode="--auto_latest_checkpoint"
attention_map_opts="" # e.g., --save_attention_map --attn_layers all --attn_mode rollout --attn_imgs_dir imgs
decoding_conf="conf/decoding/basic_decoding.json"

# training config
nj=4
gpuid=0
suffix=
train_conf="conf/macslu_qwen3_asr_17b_ep3_lora_woemblmhead.json"
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

if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding_conf not found: $decoding_conf"
    exit 1
fi

conf_tag=$(basename -s .json "$train_conf")
exp_root=${exp_root}/${conf_tag}${suffix}
phase1_exp_dir=${exp_root}/phase1_no_single
phase2_exp_dir=${exp_root}/phase2_no_single_double
phase3_exp_dir=${exp_root}/phase3_all
no_single_dir=${curriculum_json_root}/no_single
no_single_double_dir=${curriculum_json_root}/no_single_double

require_jsonl_split() {
    local split=$1
    local path=${json_root}/${split}.jsonl

    if [ ! -f "$path" ]; then
        echo "[ERROR] Required jsonl not found: $path"
        echo "[ERROR] Please prepare fixed MAC-SLU jsonl first, e.g.: ./run_macslu_fixed.sh --stage 1 --stop_stage 2"
        exit 1
    fi
}

latest_checkpoint() {
    local dir=$1

    if [ ! -d "$dir" ]; then
        echo "[ERROR] checkpoint directory not found: $dir" >&2
        return 1
    fi

    local latest
    latest=$(find "$dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
        | awk -F- '$2 ~ /^[0-9]+$/ {print $2 " " $0}' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-)

    if [ -z "$latest" ]; then
        echo "[ERROR] no checkpoint-* directory found under: $dir" >&2
        return 1
    fi

    echo "${dir}/${latest}"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Prepare MAC-SLU curriculum jsonl subsets"

    require_jsonl_split train
    require_jsonl_split dev
    require_jsonl_split test

    python local/filter_macslu_by_semantics_len.py \
        --jsonl-root "$json_root" \
        --output-dir "$no_single_dir" \
        --max-semantics-len 1 \
        --splits train dev

    python local/filter_macslu_by_semantics_len.py \
        --jsonl-root "$json_root" \
        --output-dir "$no_single_double_dir" \
        --max-semantics-len 2 \
        --splits train dev
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Finetuning on no intent + single intent MAC-SLU"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed "$seed" \
            --train_conf "$train_conf" \
            --train_file "${no_single_dir}/train.jsonl" \
            --eval_file "${no_single_dir}/dev.jsonl" \
            --output_dir "$phase1_exp_dir"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Finetuning on no intent + single intent + double intent MAC-SLU"

    phase1_checkpoint=$(latest_checkpoint "$phase1_exp_dir")
    echo "[INFO] Warm-start from phase 1 checkpoint: $phase1_checkpoint"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed "$seed" \
            --init_from_checkpoint "$phase1_checkpoint" \
            --train_conf "$train_conf" \
            --train_file "${no_single_double_dir}/train.jsonl" \
            --eval_file "${no_single_double_dir}/dev.jsonl" \
            --output_dir "$phase2_exp_dir"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Finetuning on all MAC-SLU intents"

    phase2_checkpoint=$(latest_checkpoint "$phase2_exp_dir")
    echo "[INFO] Warm-start from phase 2 checkpoint: $phase2_checkpoint"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py --seed "$seed" \
            --init_from_checkpoint "$phase2_checkpoint" \
            --train_conf "$train_conf" \
            --train_file "${json_root}/train.jsonl" \
            --eval_file "${json_root}/dev.jsonl" \
            --output_dir "$phase3_exp_dir"
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Inference on MAC-SLU test with final curriculum model"

    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl

        mkdir -p ${phase3_exp_dir}/${test_set}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir "$phase3_exp_dir" \
                --input_jsonl "$test_jsonl" \
                --output_root "$phase3_exp_dir" \
                --device cuda:0 \
                --decoding_conf "$decoding_conf" \
                $attention_map_opts
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Evaluate MAC-SLU predictions from final curriculum model"

    for test_set in $test_sets; do
        pred_file=${phase3_exp_dir}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/metrics.py --output_dir "${phase3_exp_dir}/${test_set}" "$pred_file" "$gt_file" \
            | tee "${phase3_exp_dir}/${test_set}/metrics.txt"
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Plot MAC-SLU confusion matrices for final curriculum model"

    for test_set in $test_sets; do
        pred_file=${phase3_exp_dir}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        output_dir=${phase3_exp_dir}/${test_set}

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        if [ ! -f "$gt_file" ]; then
            echo "[WARNING] ground truth file not found: $gt_file"
            continue
        fi

        python local/plot_macslu_confusion.py \
            --pred_file "$pred_file" \
            --gt_file "$gt_file" \
            --labels_file "${data_root}/labels.txt" \
            --label_mapping_file "${data_root}/labels_zh_en.txt" \
            --output_dir "$output_dir"
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Summary (MAC-SLU curriculum final model)"

    for test_set in $test_sets; do
        metrics_file=${phase3_exp_dir}/${test_set}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi

        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi
