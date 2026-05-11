#!/bin/bash

set -euo pipefail

# config
train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead.json"
output_jsonl_dir="data-json/macslu_2nd"
corrected_prompt_file="data/macslu/corrected_prompt.txt"

gpuid=0
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

conf_tag=$(basename -s .json $train_conf)
src_exp_dir=exp/macslu/${conf_tag}
exp_root=exp/$(basename "$output_jsonl_dir")/

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Generate corrected jsonl for train/dev/test"

    for split in train dev test; do
        input_jsonl=data-json/macslu/${split}.jsonl

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                --auto_latest_checkpoint \
                --exp_dir "$src_exp_dir" \
                --input_jsonl "$input_jsonl" \
                --output_root "$src_exp_dir" \
                --device cuda:0 \
                --output_jsonl_dir "$output_jsonl_dir" \
                --corrected_prompt_file "$corrected_prompt_file"
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Run 2nd training via run_macslu.sh"

    ./run_macslu.sh \
        --stage 1 \
        --train_conf $train_conf \
        --gpuid $gpuid \
        --data_root "$output_jsonl_dir" \
        --exp_root "$exp_root"
fi

checkpoint=exp/macslu/macslu_qwen3_asr_17b_ep10_lora_woemblmhead/checkpoint-1400
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Run 2nd training via run_macslu.sh ($checkpoint)"

    ./run_macslu.sh \
        --stage 1 \
        --train_conf $train_conf \
        --gpuid $gpuid \
        --data_root "$output_jsonl_dir" \
        --exp_root "${exp_root}_rsume" \
        --checkpoint $checkpoint
fi
