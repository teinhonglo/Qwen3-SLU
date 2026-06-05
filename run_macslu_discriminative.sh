#!/bin/bash
# Discriminative MAC-SLU recipe: generate same-audio n-best hypotheses from
# a source model, train with an MMI objective, then reuse run_macslu.sh testing.

set -euo pipefail

# data config (kept aligned with run_macslu.sh)
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
json_root="data-json/macslu"
nbest_json_root="data-json/macslu_mmi"
exp_root="exp/macslu_discriminative"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
prompt_file=""
attention_map_opts=""
decoding_conf="conf/decoding/basic_decoding.json"
nbest_decoding_conf="conf/decoding/nbest_decoding.json"
inference_mode="--auto_latest_checkpoint"

# source model used to create denominator n-best hypotheses and initialize MMI training
src_model=""
mmi_init_model=""
mmi_init_checkpoint_mode="latest"  # latest, best, or none
mmi_train_conf=""  # default: source model train_conf.json, then train_conf

# training config
nj=4
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_06b_ep10_lora.json
seed=66
checkpoint=

# stage
stage=0
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ -z "$mmi_init_model" ] && [ -n "$src_model" ]; then
    mmi_init_model="$src_model"
fi

if [ -z "$mmi_train_conf" ] && [ -n "$mmi_init_model" ] && [ -f "$mmi_init_model/train_conf.json" ]; then
    mmi_train_conf="$mmi_init_model/train_conf.json"
fi
if [ -z "$mmi_train_conf" ]; then
    mmi_train_conf="$train_conf"
fi

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi

if [ ! -f "$mmi_train_conf" ]; then
    echo "[ERROR] mmi_train_conf not found: $mmi_train_conf"
    exit 1
fi

if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding_conf not found: $decoding_conf"
    exit 1
fi

if [ ! -f "$nbest_decoding_conf" ]; then
    echo "[ERROR] nbest_decoding_conf not found: $nbest_decoding_conf"
    exit 1
fi

conf_tag=$(basename -s .json "$mmi_train_conf")
exp_base=$exp_root
exp_dir=${exp_base}/${conf_tag}${suffix}

if [ "$checkpoint" != "" ]; then
    training_opts="--resume_from $checkpoint --resume 1"
else
    training_opts=""
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Verify MAC-SLU jsonl already exists"

    missing=0
    for split in train dev test; do
        f=${json_root}/${split}.jsonl
        if [ ! -f "$f" ]; then
            echo "[ERROR] missing required file: $f"
            missing=1
        fi
    done

    if [ "$missing" -ne 0 ]; then
        echo "[HINT] Please prepare MAC-SLU first by running run_macslu.sh stage 0, e.g.:"
        echo "       ./run_macslu.sh --stage 0 --stop_stage 0 --json_root $json_root"
        exit 1
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate same-audio n-best denominator JSONL from src_model"

    if [ -z "$src_model" ]; then
        echo "[ERROR] --src_model is required for discriminative training"
        exit 1
    fi

    for split in train dev; do
        input_jsonl=${json_root}/${split}.jsonl
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            echo "[HINT] Please prepare MAC-SLU first by running run_macslu.sh stage 0, e.g.:"
            echo "       ./run_macslu.sh --stage 0 --stop_stage 0 --json_root $json_root"
            exit 1
        fi

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir "$src_model" \
                --input_jsonl "$input_jsonl" \
                --output_root "$src_model" \
                --device cuda:0 \
                --decoding_conf "$nbest_decoding_conf" \
                --output_nbest_jsonl_dir "$nbest_json_root"
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: MMI discriminative finetuning on MAC-SLU"

    if [ -z "$mmi_init_model" ]; then
        echo "[ERROR] --src_model or --mmi_init_model is required so MMI starts from an existing model"
        exit 1
    fi

    init_opts=(--init_model_dir "$mmi_init_model")
    if [ "$mmi_init_checkpoint_mode" = "latest" ]; then
        init_opts+=(--auto_latest_init_checkpoint)
    elif [ "$mmi_init_checkpoint_mode" = "best" ]; then
        init_opts+=(--auto_best_init_checkpoint)
    elif [ "$mmi_init_checkpoint_mode" != "none" ]; then
        echo "[ERROR] unsupported mmi_init_checkpoint_mode: $mmi_init_checkpoint_mode (expected latest, best, or none)"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_mmi.py --seed $seed $training_opts \
            "${init_opts[@]}" \
            --train_conf "$mmi_train_conf" \
            --train_file "$nbest_json_root/train.jsonl" \
            --eval_file "$nbest_json_root/dev.jsonl" \
            --output_dir "$exp_dir"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3-5: Reuse run_macslu.sh test/eval/summary"

    ./run_macslu.sh \
        --stage 2 \
        --stop_stage 4 \
        --json_root "$json_root" \
        --exp_root "$exp_base" \
        --suffix "$suffix" \
        --train_conf "$mmi_train_conf" \
        --gpuid "$gpuid" \
        --test_sets "$test_sets" \
        --inference_mode "$inference_mode" \
        --attention_map_opts "$attention_map_opts" \
        --decoding_conf "$decoding_conf"
fi
