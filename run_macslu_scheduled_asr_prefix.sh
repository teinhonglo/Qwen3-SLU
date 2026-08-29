#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
audio_dir=${data_root}/audio
labels_file="data/macslu/labels.txt"
label_mapping_file="data/macslu/labels_zh_en.txt"
json_root="data-json/macslu_fixed"
exp_root="exp/macslu_fixed"
inference_mode="--auto_latest_checkpoint"
prompt_file=""   # 可指定外部 prompt 檔案，空字串則使用 prepare_macslu_jsonl.py 內建 prompt
attention_map_opts="" # e.g., --save_attention_map --attn_layers all --attn_mode rollout --attn_imgs_dir imgs
decoding_conf="conf/decoding/basic_decoding.json"

# training config
nj=4
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead_scheduled_asr_prefix.json
seed=66
checkpoint=
asr_batch_size=16
asr_language=Chinese

# stage
stage=1
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

conf_tag=$(basename -s .json $train_conf)
decoding_conf_name=$(basename -s .json "$decoding_conf")
exp_root=$exp_root/${conf_tag}${suffix}

if [ "$checkpoint" != "" ]; then
    training_opts="--resume_from $checkpoint --resume 1"
else
    training_opts=""
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download MAC-SLU and prepare jsonl"

    prep_cmd=(
        python local/prepare_macslu_jsonl.py
        --repo-id "$repo_id"
        --download-dir "$download_dir"
        --extract-root "$extract_root"
        --jsonl-root "$json_root"
        --splits train dev test
    )

    if [ -n "$prompt_file" ]; then
        prep_cmd+=(--prompt-file "$prompt_file")
    fi

    "${prep_cmd[@]}"

    python local/count_macslu_intent_distribution.py \
        --jsonl-root "$json_root" \
        --splits train dev test \
        --output-txt "${json_root}/intent_distribution.txt" \
        --output-json "${json_root}/intent_distribution.json"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate scheduled-ASR-prefix ASR targets"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python local/prepare_macslu_scheduled_asr_prefix.py \
            --train_conf "$train_conf" \
            --input_jsonl "$json_root/train.jsonl" \
            --output_jsonl "$json_root/train_scheduled_asr_prefix.jsonl" \
            --device cuda:0 \
            --language "$asr_language" \
            --batch_size "$asr_batch_size"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Finetuning on MAC-SLU with weighted generated-ASR loss"

    data_dir=$json_root
    exp_dir=$exp_root

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft_scheduled_asr_prefix.py --seed $seed $training_opts \
            --train_conf $train_conf \
            --train_file $data_dir/train_scheduled_asr_prefix.jsonl \
            --eval_file $data_dir/dev.jsonl \
            --output_dir $exp_dir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Inference on MAC-SLU test"

    data_dir=$json_root
    exp_dir=$exp_root

    for test_set in $test_sets; do
        test_jsonl=${data_dir}/${test_set}.jsonl

        mkdir -p ${exp_dir}/${test_set}_${decoding_conf_name}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir $exp_dir \
                --input_jsonl $test_jsonl \
                --output_root $exp_dir \
                --device cuda:0 \
                --decoding_conf $decoding_conf \
                $attention_map_opts
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Evaluate MAC-SLU predictions"

    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}_${decoding_conf_name}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/metrics.py --output_dir ${exp_root}/${test_set}_${decoding_conf_name} "$pred_file" "$gt_file" | tee ${exp_root}/${test_set}_${decoding_conf_name}/metrics.txt
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Plot MAC-SLU evaluation charts"

    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}_${decoding_conf_name}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        output_dir=${exp_root}/${test_set}_${decoding_conf_name}

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
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Summary (MAC-SLU)"

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
