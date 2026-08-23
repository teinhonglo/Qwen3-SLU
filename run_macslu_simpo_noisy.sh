#!/bin/bash
# Noisy SimPO MAC-SLU recipe: generate same-audio n-best hypotheses, score them with
# oracle/SLU metrics, build chosen/rejected preference pairs, export analysis,
# train SimPO, then reuse run_macslu.sh testing/evaluation.

set -euo pipefail

# data config (kept aligned with run_macslu.sh)
json_root="data-json/macslu_fixed"
exp_root="exp/macslu_simpo"
noise_dir="/share/corpus/aishell5/noise"
snr_db=15
noise_seed=42
attention_map_opts=""
decoding_conf="conf/decoding/nbest_decoding.json"
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
# nbest_only requires a generated oracle; nbest_oracle falls back to the ground truth.
# oracle_balance also uses GT fallback, while deterministically retaining all rank-0
# errors and 2.5%/15%/20%/40% of rank-0-correct samples with 0/1/2/3+ intents.
pair_mode="nbest_oracle"
pair_min_score_margin="0.1"
pair_max_pairs_per_sample="1"
# Generate and score test n-best for analysis, but keep pair/training splits to train/dev to avoid test leakage.
nbest_splits="train dev test"
score_splits="train dev test"
pair_splits="train dev"
# Export sample-level n-best/pair analysis for the same splits used to build pairs.
analysis_splits="train dev"

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
nbest_decoding_conf_name=$(basename -s .json "$nbest_decoding_conf")
# A filesystem-safe tag separates every noisy dataset and experiment.
snr_tag=$(printf '%s' "$snr_db" | sed -e 's/^-//; s/\./p/g')
case "$snr_db" in -*) snr_tag="m${snr_tag}" ;; esac
if ! [[ "$snr_tag" =~ ^m?[0-9]+(p[0-9]+)?$ ]]; then
    echo "[ERROR] --snr_db must be a single finite decimal number: $snr_db"
    exit 1
fi
noisy_tag="noisy_snr${snr_tag}_seed${noise_seed}"
noisy_root="${json_root}/${noisy_tag}"
noisy_audio_dir="${noisy_root}/audio"
artifact_root="${exp_root}_${noisy_tag}/nbest_from_sft"
noisy_conf_dir="${exp_root}_${noisy_tag}/config"
noisy_simpo_train_conf="${noisy_conf_dir}/$(basename "$simpo_train_conf")"
# Keep runs made with different pair construction policies in separate trees.
exp_base=${exp_root}_${noisy_tag}_${pair_mode}
exp_dir=${exp_base}/${conf_tag}${suffix}

if [ "$checkpoint" != "" ]; then
    training_opts="--resume_from $checkpoint --resume 1"
else
    training_opts=""
fi

# Stage 0: Create noisy training audio/JSONL; clean dev and test remain unchanged.
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Add AISHELL-5 in-car background noise to MAC-SLU train"
    clean_train_jsonl="${json_root}/train.jsonl"
    noisy_train_jsonl="${noisy_root}/train.jsonl"
    if [ ! -f "$clean_train_jsonl" ]; then
        echo "[ERROR] missing clean train JSONL: $clean_train_jsonl"
        exit 1
    fi
    if [ ! -d "$noise_dir" ]; then
        echo "[ERROR] noise_dir does not exist: $noise_dir"
        exit 1
    fi
    if [ -s "$noisy_train_jsonl" ]; then
        if [ ! -d "$noisy_audio_dir" ]; then
            echo "[ERROR] noisy JSONL exists but noisy audio directory is missing: $noisy_audio_dir"
            exit 1
        fi
        if [ ! -s "${noisy_train_jsonl}.noise_meta.jsonl" ]; then
            echo "[ERROR] noisy JSONL exists but metadata sidecar is missing or empty: ${noisy_train_jsonl}.noise_meta.jsonl"
            exit 1
        fi
        echo "[SKIP] Existing non-empty noisy train JSONL: $noisy_train_jsonl"
    else
        python local/add_noise_to_jsonl.py \
            --input_jsonl "$clean_train_jsonl" \
            --output_jsonl "$noisy_train_jsonl" \
            --output_audio_dir "$noisy_audio_dir" \
            --noise_dir "$noise_dir" \
            --snr_db "$snr_db" \
            --seed "$noise_seed"
    fi
fi

# Stage 1: Use the clean SFT checkpoint to generate run-isolated n-best JSONL.
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate n-best JSONL from src_exp_dir for: $nbest_splits"

    if [ -z "$src_exp_dir" ]; then
        echo "[ERROR] --src_exp_dir is required for SimPO preference data generation"
        exit 1
    fi

    for split in $nbest_splits; do
        if [ "$split" = train ]; then
            input_jsonl=${noisy_root}/train.jsonl
        else
            input_jsonl=${json_root}/${split}.jsonl
        fi
        pred_file=${artifact_root}/${split}_${nbest_decoding_conf_name}/predictions.jsonl
        gt_file=${input_jsonl}
        
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            exit 1
        fi
        
        if [ ! -f "$pred_file" ]; then
        
            output_nbest_dir="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest"
            
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python finetuning/qwen3_asr_test.py \
                    $inference_mode \
                    --exp_dir "$src_exp_dir" \
                    --input_jsonl "$input_jsonl" \
                    --output_root "$artifact_root" \
                    --device cuda:0 \
                    --decoding_conf "$nbest_decoding_conf" \
                    --output_nbest_jsonl_dir "$output_nbest_dir"
        else
            echo "Existed file (Evaluation Only): $pred_file"
            python local/metrics.py --output_dir ${artifact_root}/${split}_${nbest_decoding_conf_name} "$pred_file" "$gt_file" | tee ${artifact_root}/${split}_${nbest_decoding_conf_name}/metrics.txt
        fi
    done
fi

# Stage 2: Score each n-best hypothesis and the ground-truth fallback candidate.
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Score n-best with oracle EMA and local/metrics.py metrics for: $score_splits"

    for split in $score_splits; do
        input_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/${split}.jsonl"
        output_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/scored_nbest.jsonl"
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
        input_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/scored_nbest.jsonl"
        output_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/simpo_pairs_${pair_mode}.jsonl"
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

# Stage 4: Export rank-aligned n-best and chosen/rejected pair data for analysis.
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Export SimPO analysis JSONL for: $analysis_splits"

    for split in $analysis_splits; do
        input_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/scored_nbest.jsonl"
        pairs_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/simpo_pairs_${pair_mode}.jsonl"
        output_jsonl="${artifact_root}/${split}_${nbest_decoding_conf_name}/nbest/simpo_analysis_${pair_mode}.jsonl"
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing required file: $input_jsonl"
            exit 1
        fi
        if [ ! -f "$pairs_jsonl" ]; then
            echo "[ERROR] missing required file: $pairs_jsonl"
            exit 1
        fi
        python local/export_simpo_analysis.py \
            --input_jsonl "$input_jsonl" \
            --pairs_jsonl "$pairs_jsonl" \
            --output_jsonl "$output_jsonl"
    done
fi

# Stage 5: Run SimPO finetuning from simpo_init_model using train/dev pairs.
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: SimPO preference finetuning on MAC-SLU"

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

    mkdir -p "$noisy_conf_dir"
    if [ ! -f "$noisy_simpo_train_conf" ]; then
        cp "$simpo_train_conf" "$noisy_simpo_train_conf"
    fi
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_simpo.py --seed $seed $training_opts \
            "${init_opts[@]}" \
            --train_conf "$noisy_simpo_train_conf" \
            --train_file "${artifact_root}/train_${nbest_decoding_conf_name}/nbest/simpo_pairs_${pair_mode}.jsonl" \
            --eval_file "${artifact_root}/dev_${nbest_decoding_conf_name}/nbest/simpo_pairs_${pair_mode}.jsonl" \
            --output_dir "$exp_dir"
fi

# Stage 6: Reuse run_macslu.sh to run standard test inference/eval/summary.
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Reuse run_macslu.sh test/eval/summary"

    mkdir -p "$noisy_conf_dir"
    if [ ! -f "$noisy_simpo_train_conf" ]; then cp "$simpo_train_conf" "$noisy_simpo_train_conf"; fi

    ./run_macslu.sh \
        --stage 2 \
        --stop_stage 4 \
        --json_root "$json_root" \
        --exp_root "$exp_base" \
        --suffix "$suffix" \
        --train_conf "$noisy_simpo_train_conf" \
        --gpuid "$gpuid" \
        --test_sets "$test_sets" \
        --inference_mode "$inference_mode" \
        --attention_map_opts "$attention_map_opts" \
        --decoding_conf "$decoding_conf"
fi
