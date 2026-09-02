#!/bin/bash
# Full RobustGER MAC-SLU recipe:
# clean data -> noisy train/dev/test -> N-best (N=10) -> SBERT/audio features
# -> two-stage RobustGER adapter training -> noisy test H2T evaluation.
#
# The existing run_macslu_simpo_noisy.sh is intentionally left unchanged.
set -euo pipefail

json_root="data-json/macslu_fixed"
exp_root="exp/macslu_robustger"
noise_dir="/share/corpus/aishell5/noise"
snr_db=5
noise_seed=42
src_exp_dir="exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead"
inference_mode="--auto_latest_checkpoint"
nbest_decoding_conf="conf/decoding/nbest_decoding.json"
robustger_conf="conf/robustger_qwen3_06b.json"
eval_train_conf="conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead.json"
gpuid=0
seed=66
stage=0
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$robustger_conf" ]; then
    echo "[ERROR] RobustGER config not found: $robustger_conf"
    exit 1
fi
if [ ! -f "$nbest_decoding_conf" ]; then
    echo "[ERROR] N-best decoding config not found: $nbest_decoding_conf"
    exit 1
fi

snr_tag=$(printf '%s' "$snr_db" | sed -e 's/^-//; s/\./p/g')
case "$snr_db" in
    -*) snr_tag="m${snr_tag}" ;;
esac
if ! [[ "$snr_tag" =~ ^m?[0-9]+(p[0-9]+)?$ ]]; then
    echo "[ERROR] --snr_db must be a finite decimal number: $snr_db"
    exit 1
fi
noise_tag="noisy_snr${snr_tag}"
nbest_conf_name=$(basename -s .json "$nbest_decoding_conf")
run_dir="${exp_root}/n10_${noise_tag}"
feature_root="${run_dir}/features"
model_dir="${run_dir}/model"
eval_output_dir="${run_dir}/test_${nbest_conf_name}"

run_macslu_eval_opts=(
    --json_root "$json_root"
    --eval_output_dir "$eval_output_dir"
    --train_conf "$eval_train_conf"
    --gpuid "$gpuid"
    --test_sets "$test_sets"
    --inference_mode "$inference_mode"
    --decoding_conf "$nbest_decoding_conf"
)

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Prepare noisy train/dev/test at SNR=${snr_db} dB"
    if [ ! -d "$noise_dir" ]; then
        echo "[ERROR] noise_dir does not exist: $noise_dir"
        exit 1
    fi

    for split in train dev test; do
        clean_jsonl="${json_root}/${split}.jsonl"
        noisy_jsonl="${json_root}/${split}_${noise_tag}.jsonl"
        audio_dir="${json_root}/audio_${noise_tag}/${split}"
        if [ ! -f "$clean_jsonl" ]; then
            echo "[ERROR] missing clean JSONL: $clean_jsonl"
            exit 1
        fi
        if [ -s "$noisy_jsonl" ]; then
            if [ ! -d "$audio_dir" ] || [ ! -s "${noisy_jsonl}.noise_meta.jsonl" ]; then
                echo "[ERROR] existing noisy JSONL is missing audio or metadata: $noisy_jsonl"
                exit 1
            fi
            echo "[SKIP] Existing noisy $split JSONL: $noisy_jsonl"
        else
            mkdir -p "$audio_dir"
            python local/add_noise_to_jsonl.py \
                --input_jsonl "$clean_jsonl" \
                --output_jsonl "$noisy_jsonl" \
                --output_audio_dir "$audio_dir" \
                --noise_dir "$noise_dir" \
                --snr_db "$snr_db" \
                --seed "$noise_seed"
        fi
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Generate N-best hypotheses (N=10) for noisy train/dev/test"
    if [ ! -d "$src_exp_dir" ]; then
        echo "[ERROR] source ASR experiment not found: $src_exp_dir"
        exit 1
    fi
    for split in train dev test; do
        tagged_split="${split}_${noise_tag}"
        input_jsonl="${json_root}/${tagged_split}.jsonl"
        output_nbest_dir="${src_exp_dir}/${tagged_split}_${nbest_conf_name}/nbest"
        nbest_file="${output_nbest_dir}/${tagged_split}.jsonl"
        if [ ! -f "$input_jsonl" ]; then
            echo "[ERROR] missing noisy JSONL: $input_jsonl"
            exit 1
        fi
        if [ -s "$nbest_file" ]; then
            echo "[SKIP] Existing N-best file: $nbest_file"
        else
            mkdir -p "$output_nbest_dir"
            CUDA_VISIBLE_DEVICES="$gpuid" python finetuning/qwen3_asr_test.py \
                $inference_mode \
                --exp_dir "$src_exp_dir" \
                --input_jsonl "$input_jsonl" \
                --output_root "$src_exp_dir" \
                --device cuda:0 \
                --decoding_conf "$nbest_decoding_conf" \
                --output_nbest_jsonl_dir "$output_nbest_dir"
        fi
        if [ ! -s "$nbest_file" ]; then
            echo "[ERROR] N-best output not found: $nbest_file"
            exit 1
        fi
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Build multilingual-SBERT language noise and clean/noisy audio features"
    for split in train dev test; do
        tagged_split="${split}_${noise_tag}"
        nbest_file="${src_exp_dir}/${tagged_split}_${nbest_conf_name}/nbest/${tagged_split}.jsonl"
        feature_dir="${feature_root}/${split}"
        if [ ! -s "$nbest_file" ]; then
            echo "[ERROR] missing N-best file: $nbest_file"
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$gpuid" python local/prepare_robustger_data.py \
            --nbest_jsonl "$nbest_file" \
            --clean_jsonl "${json_root}/${split}.jsonl" \
            --noisy_jsonl "${json_root}/${tagged_split}.jsonl" \
            --output_dir "$feature_dir" \
            --config "$robustger_conf" \
            --device cuda:0
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Train full RobustGER adapter and MINE on RTX 3090"
    CUDA_VISIBLE_DEVICES="$gpuid" python finetuning/train_qwen3_robustger.py \
        --train_conf "$robustger_conf" \
        --train_file "${feature_root}/train/manifest.jsonl" \
        --eval_file "${feature_root}/dev/manifest.jsonl" \
        --output_dir "$model_dir" \
        --device cuda:0 \
        --seed "$seed"
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Decode noisy test with RobustGER"
    test_feature_dir="${feature_root}/test"
    prediction_file="${run_dir}/test_${nbest_conf_name}/predictions.jsonl"
    if [ ! -s "${model_dir}/adapter-best.pt" ]; then
        echo "[ERROR] RobustGER checkpoint not found: ${model_dir}/adapter-best.pt"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="$gpuid" python finetuning/test_qwen3_robustger.py \
        --manifest "${test_feature_dir}/manifest.jsonl" \
        --checkpoint "${model_dir}/adapter-best.pt" \
        --config "$robustger_conf" \
        --output_jsonl "$prediction_file" \
        --device cuda:0
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Evaluate RobustGER via run_macslu.sh"
    ./run_macslu.sh --stage 3 --stop_stage 3 "${run_macslu_eval_opts[@]}"
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Plot RobustGER evaluation via run_macslu.sh"
    ./run_macslu.sh --stage 4 --stop_stage 4 "${run_macslu_eval_opts[@]}"
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Summarize RobustGER evaluation via run_macslu.sh"
    ./run_macslu.sh --stage 5 --stop_stage 5 "${run_macslu_eval_opts[@]}"
fi
