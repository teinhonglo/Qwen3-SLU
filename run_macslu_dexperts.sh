#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
json_root=data-json/macslu
dexperts_root=data-json/macslu_dexperts
exp_root=exp/macslu
inference_mode="--auto_latest_checkpoint"

# training / inference config
nj=4
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead.json
seed=66
expert_train_conf=conf/expert/qwen3_text_expert_17b.json

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

if [ ! -f "$expert_train_conf" ]; then
    echo "[ERROR] expert_train_conf not found: $expert_train_conf"
    exit 1
fi

conf_tag=$(basename -s .json $train_conf)
base_exp_dir=${exp_root}/${conf_tag}${suffix}
expert_conf_tag=$(basename -s .json $expert_train_conf)
expert_exp_root=$base_exp_dir/dexperts_experts_${expert_conf_tag}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Verify legacy MAC-SLU jsonl exists"
    for split in train dev test; do
        f=${json_root}/${split}.jsonl
        if [ ! -f "$f" ]; then
            echo "[ERROR] missing required file: $f"
            echo "[HINT] Please run run_macslu.sh stage 0 first."
            exit 1
        fi
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Build DExperts schema"
    python local/build_macslu_schema.py \
        --input_jsonls ${json_root}/train.jsonl ${json_root}/dev.jsonl \
        --output_json ${dexperts_root}/schema.json
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Build DExperts expert corpora"
    python local/build_macslu_dexperts_data.py \
        --train_jsonl ${json_root}/train.jsonl \
        --dev_jsonl ${json_root}/dev.jsonl \
        --output_dir ${dexperts_root}
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Train domain-intent expert"
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/train_expert_lm.py \
            --train_jsonl ${dexperts_root}/domain_intent_train.jsonl \
            --dev_jsonl ${dexperts_root}/domain_intent_dev.jsonl \
            --train_conf ${expert_train_conf} \
            --output_dir ${expert_exp_root}/domain_intent
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Train slot-key expert"
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/train_expert_lm.py \
            --train_jsonl ${dexperts_root}/slot_key_train.jsonl \
            --dev_jsonl ${dexperts_root}/slot_key_dev.jsonl \
            --train_conf ${expert_train_conf} \
            --output_dir ${expert_exp_root}/slot_key
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: DExperts inference on MAC-SLU test"

    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl
        mkdir -p ${expert_exp_root}/${test_set}

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test_dexperts.py \
                $inference_mode \
                --exp_dir ${base_exp_dir} \
                --input_jsonl ${test_jsonl} \
                --output_root ${expert_exp_root} \
                --device cuda:0 \
                --use_dexperts \
                --dexperts_config conf/decoding/dexperts_macslu.json \
                --schema_path ${dexperts_root}/schema.json \
                --domain_intent_expert_path ${expert_exp_root}/domain_intent \
                --slot_key_expert_path ${expert_exp_root}/slot_key
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Evaluate MAC-SLU predictions (DExperts)"

    for test_set in $test_sets; do
        pred_file=${expert_exp_root}/${test_set}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        python local/metrics.py --output_dir ${expert_exp_root}/${test_set} "$pred_file" "$gt_file" | tee ${expert_exp_root}/${test_set}/metrics.txt
    done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "Stage 8: Summary (MAC-SLU DExperts)"

    for test_set in $test_sets; do
        metrics_file=${expert_exp_root}/${test_set}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi

        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi
