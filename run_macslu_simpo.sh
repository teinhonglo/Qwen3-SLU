#!/bin/bash
# SimPO MAC-SLU recipe: generate same-audio n-best hypotheses, score them with
# oracle/SLU metrics, build chosen/rejected preference pairs, train SimPO, then
# reuse run_macslu.sh testing/evaluation.

set -euo pipefail

# data config (kept aligned with run_macslu.sh)
json_root="data-json/macslu"
exp_root="exp/macslu_simpo"
attention_map_opts=""
decoding_conf="conf/decoding/basic_decoding.json"
nbest_decoding_conf="conf/decoding/nbest_decoding.json"
inference_mode="--auto_latest_checkpoint"

# source model used to create n-best hypotheses and initialize SimPO training
src_exp_dir="exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead"
# Backward-compatible alias; prefer --src_exp_dir for new runs.
src_model=""
simpo_init_model=""
simpo_init_checkpoint_mode="latest"  # latest, best, or none
simpo_train_conf=""  # default: SimPO paper-style train_conf

# SimPO trainer hyperparameters live in conf/*simpo.json.
# Pair-building settings are pipeline controls for local/build_simpo_pairs.py.
pair_mode="nbest_only"
pair_min_score_margin="0.1"
pair_max_pairs_per_sample="1"
# Generate and score test n-best for analysis, but keep pair/training splits to train/dev to avoid test leakage.
nbest_splits="train dev test"
score_splits="train dev test"
pair_splits="train dev"

# training config
gpuid=0
suffix=
train_conf="conf/macslu_qwen3_asr_simpo.json"
seed=66
checkpoint=

# stage
stage=0
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ -z "$src_exp_dir" ] && [ -n "$src_model" ]; then
    src_exp_dir="$src_model"
fi

if [ -z "$simpo_init_model" ] && [ -n "$src_exp_dir" ]; then
    simpo_init_model="$src_exp_dir"
fi

if [ -z "$simpo_train_conf" ]; then
    simpo_train_conf="$train_conf"
fi

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi

if [ ! -f "$simpo_train_conf" ]; then
    echo "[ERROR] simpo_train_conf not found: $simpo_train_conf"
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

conf_tag=$(basename -s .json "$simpo_train_conf")
exp_base=$exp_root
exp_dir=${exp_base}/${conf_tag}${suffix}

nbest_dir_for_split() {
    echo "${src_exp_dir}/$1/nbest"
}

nbest_jsonl_for_split() {
    echo "$(nbest_dir_for_split "$1")/$1.jsonl"
}

scored_jsonl_for_split() {
    echo "$(nbest_dir_for_split "$1")/scored_nbest.jsonl"
}

pair_jsonl_for_split() {
    echo "$(nbest_dir_for_split "$1")/simpo_pairs.jsonl"
}

if [ "$checkpoint" != "" ]; then
    training_opts="--resume_from $checkpoint --resume 1"
else
    training_opts=""
fi

# Stage 0: Check that prepared MAC-SLU train/dev/test JSONL files exist.
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

# Stage 1: Use src_exp_dir to generate n-best JSONL under src_exp_dir/<split>/nbest/.

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate same-audio n-best JSONL from src_exp_dir for: $nbest_splits"

    if [ -z "$src_exp_dir" ]; then
        echo "[ERROR] --src_exp_dir is required for SimPO preference data generation"
        exit 1
    fi

    for split in $nbest_splits; do
        input_jsonl=${json_root}/${split}.jsonl
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            exit 1
        fi

        output_nbest_dir=$(nbest_dir_for_split "$split")

        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir "$src_exp_dir" \
                --input_jsonl "$input_jsonl" \
                --output_root "$src_exp_dir" \
                --device cuda:0 \
                --decoding_conf "$nbest_decoding_conf" \
                --output_nbest_jsonl_dir "$output_nbest_dir"
    done
fi

# Stage 2: Score each n-best hypothesis under src_exp_dir/<split>/nbest/.

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Score n-best with oracle EMA and local/metrics.py metrics for: $score_splits"

    for split in $score_splits; do
        input_jsonl=$(nbest_jsonl_for_split "$split")
        output_jsonl=$(scored_jsonl_for_split "$split")
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            exit 1
        fi
        python local/score_nbest_oracle.py \
            --input_jsonl "$input_jsonl" \
            --output_jsonl "$output_jsonl"
    done
fi

# Stage 3: Build chosen/rejected SimPO pairs under src_exp_dir/<split>/nbest/.

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Build SimPO chosen/rejected pairs for: $pair_splits"

    for split in $pair_splits; do
        input_jsonl=$(scored_jsonl_for_split "$split")
        output_jsonl=$(pair_jsonl_for_split "$split")
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            exit 1
        fi
        python local/build_simpo_pairs.py \
            --input_jsonl "$input_jsonl" \
            --output_jsonl "$output_jsonl" \
            --pair_mode "$pair_mode" \
            --min_score_margin "$pair_min_score_margin" \
            --max_pairs_per_sample "$pair_max_pairs_per_sample"
    done
fi

# Stage 4: Run SimPO finetuning from simpo_init_model using train/dev pairs.
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: SimPO preference finetuning on MAC-SLU"

    if [ -z "$simpo_init_model" ]; then
        echo "[ERROR] --src_exp_dir or --simpo_init_model is required so SimPO starts from SFT/instruction-tuned weights"
        exit 1
    fi

    init_opts=(--init_model_dir "$simpo_init_model")
    if [ "$simpo_init_checkpoint_mode" = "latest" ]; then
        init_opts+=(--auto_latest_init_checkpoint)
    elif [ "$simpo_init_checkpoint_mode" = "best" ]; then
        init_opts+=(--auto_best_init_checkpoint)
    elif [ "$simpo_init_checkpoint_mode" != "none" ]; then
        echo "[ERROR] unsupported simpo_init_checkpoint_mode: $simpo_init_checkpoint_mode (expected latest, best, or none)"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_simpo.py --seed $seed $training_opts \
            "${init_opts[@]}" \
            --train_conf "$simpo_train_conf" \
            --train_file "$(pair_jsonl_for_split train)" \
            --eval_file "$(pair_jsonl_for_split dev)" \
            --output_dir "$exp_dir"
fi

# Stage 5: Reuse run_macslu.sh to run standard test inference/eval/summary.
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Reuse run_macslu.sh test/eval/summary"

    ./run_macslu.sh \
        --stage 2 \
        --stop_stage 4 \
        --json_root "$json_root" \
        --exp_root "$exp_base" \
        --suffix "$suffix" \
        --train_conf "$simpo_train_conf" \
        --gpuid "$gpuid" \
        --test_sets "$test_sets" \
        --inference_mode "$inference_mode" \
        --attention_map_opts "$attention_map_opts" \
        --decoding_conf "$decoding_conf"
fi
