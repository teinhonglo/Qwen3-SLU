#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
audio_dir=${data_root}/audio
json_root=${data_root}/json
prompt_file=""   # 可指定外部 prompt 檔案，空字串則使用 prepare_macslu_jsonl.py 內建 prompt

# training config
nj=4
gpuid=0
suffix=
train_conf=finetuning/train_conf/macslu_qwen3_asr_06b.json

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
exp_root=exp/macslu/${conf_tag}${suffix}

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
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Finetuning on MAC-SLU"

    data_dir=$json_root
    exp_dir=$exp_root

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft.py \
            --train_conf $train_conf \
            --train_file $data_dir/train.jsonl \
            --eval_file $data_dir/dev.jsonl \
            --output_dir $exp_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Inference on MAC-SLU test"

    data_dir=$json_root
    exp_dir=$exp_root

    for test_set in $test_sets; do
        test_jsonl=${data_dir}/${test_set}.jsonl

        mkdir -p ${exp_dir}/${test_set}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                --exp_dir $exp_dir \
                --auto_latest_checkpoint \
                --input_jsonl $test_jsonl \
                --output_root $exp_dir \
                --device cuda:0
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate MAC-SLU predictions"

    for test_set in $test_sets; do
        pred_file=${exp_root}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/metrics.py "$pred_file" "$gt_file" | tee ${exp_root}/${test_set}/metrics.txt
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Summary (MAC-SLU)"

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
