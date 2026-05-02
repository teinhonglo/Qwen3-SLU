#!/bin/bash

set -euo pipefail

# config
output_jsonl_dir="data-json/macslu_2nd"
corrected_prompt_file="data/macslu/corrected_prompt.txt"
src_exp_dir="exp/macslu/macslu_qwen3_asr_06b"
exp_root=exp/$(basename "$output_jsonl_dir")/

gpuid=0
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Generate corrected jsonl for train/dev/test"

    for split in train dev test; do
        input_jsonl=data-json/macslu/${split}.jsonl

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                --auto_latest_checkpoint \
                --exp_dir "$src_exp_dir" \
                --input_jsonl "$input_jsonl" \
                --output_root "$exp_root" \
                --device cuda:0 \
                --output_jsonl_dir "$output_jsonl_dir" \
                --corrected_prompt_file "$corrected_prompt_file"
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Run 2nd training via run_macslu.sh"

    ./run_macslu.sh \
        --stage 1 \
        --data_root "$output_jsonl_dir" \
        --exp_root "$exp_root"
fi
