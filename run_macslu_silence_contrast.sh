#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
labels_file="data/macslu/labels.txt"
label_mapping_file="data/macslu/labels_zh_en.txt"
json_root="data-json/macslu_fixed"
exp_root="exp/macslu_fixed"
inference_mode="--auto_latest_checkpoint"

# training config
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead.json

# silence contrast config
alpha=1.0
contrast_scope="both"
max_new_tokens=256
silence_value=0.0
plausibility_alpha=0.0
contrast_modes="silence first_step all_steps"

# stage
stage=2
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi

conf_tag=$(basename -s .json "$train_conf")
exp_root=$exp_root/${conf_tag}${suffix}

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Silence contrast inference on MAC-SLU test"

    data_dir=$json_root
    exp_dir=$exp_root

    for test_set in $test_sets; do
        test_jsonl=${data_dir}/${test_set}.jsonl
        output_dir=${exp_dir}/${test_set}_silence_contrast

        mkdir -p "$output_dir"

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test_silence_contrast.py \
                $inference_mode \
                --exp_dir "$exp_dir" \
                --input_jsonl "$test_jsonl" \
                --output_dir "$output_dir" \
                --device cuda:0 \
                --max_new_tokens "$max_new_tokens" \
                --alpha "$alpha" \
                --contrast_scope "$contrast_scope" \
                --silence_value "$silence_value" \
                --plausibility_alpha "$plausibility_alpha"
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate MAC-SLU silence contrast predictions"

    for test_set in $test_sets; do
        output_dir=${exp_root}/${test_set}_silence_contrast
        gt_file=${json_root}/${test_set}.jsonl

        for mode in $contrast_modes; do
            pred_file=${output_dir}/predictions_${mode}.jsonl
            mode_output_dir=${output_dir}/${mode}

            if [ ! -f "$pred_file" ]; then
                echo "[WARNING] prediction file not found: $pred_file"
                continue
            fi

            mkdir -p "$mode_output_dir"
            python local/metrics.py \
                --output_dir "$mode_output_dir" \
                "$pred_file" "$gt_file" \
                | tee "${mode_output_dir}/metrics.txt"
        done
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Plot MAC-SLU silence contrast evaluation charts"

    for test_set in $test_sets; do
        contrast_output_dir=${exp_root}/${test_set}_silence_contrast
        gt_file=${json_root}/${test_set}.jsonl

        for mode in $contrast_modes; do
            pred_file=${contrast_output_dir}/predictions_${mode}.jsonl
            output_dir=${contrast_output_dir}/${mode}

            if [ ! -f "$pred_file" ]; then
                echo "[WARNING] prediction file not found: $pred_file"
                continue
            fi

            if [ ! -f "$gt_file" ]; then
                echo "[WARNING] ground truth file not found: $gt_file"
                continue
            fi

            python local/plot_macslu_evaluation.py \
                --pred_file "$pred_file" \
                --gt_file "$gt_file" \
                --train_file "${json_root}/train.jsonl" \
                --labels_file "$labels_file" \
                --label_mapping_file "$label_mapping_file" \
                --output_dir "$output_dir"
        done
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Summary (MAC-SLU silence contrast)"

    for test_set in $test_sets; do
        output_dir=${exp_root}/${test_set}_silence_contrast

        for mode in $contrast_modes; do
            metrics_file=${output_dir}/${mode}/metrics.txt
            if [ ! -f "$metrics_file" ]; then
                echo "[WARNING] metrics file not found: $metrics_file"
                continue
            fi

            echo "========== ${test_set}: ${mode} =========="
            cat "$metrics_file"
        done
    done
fi
