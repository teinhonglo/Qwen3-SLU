#!/bin/bash
# Diagnostic for MAC-SLU semantic-frame early STOP decisions.

set -euo pipefail

# data/model config
json_root=data-json/macslu
src_exp=exp/macslu_fixed/macslu_qwen3_asr_17b_ep10_lora_woemblmhead
out_subdir=semantic_out
analysis_root=${src_exp}/semantic_out
inference_mode="--auto_latest_checkpoint"
decoding_conf="conf/decoding/basic_decoding.json"
gpuid=0
device="cuda:0"
test_sets="dev test"
overwrite=false
limit=0
smoke_print=10

# stage
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Check MAC-SLU dev/test JSONL and source experiment"
    if [ ! -d "$src_exp" ]; then
        echo "[ERROR] src_exp not found: $src_exp"
        exit 1
    fi
    if [ ! -f "${src_exp}/train_conf.json" ]; then
        echo "[ERROR] train_conf.json not found under src_exp: ${src_exp}/train_conf.json"
        exit 1
    fi
    if [ ! -f "$decoding_conf" ]; then
        echo "[ERROR] decoding_conf not found: $decoding_conf"
        exit 1
    fi
    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl
        if [ ! -f "$test_jsonl" ]; then
            echo "[ERROR] input JSONL not found: $test_jsonl"
            exit 1
        fi
    done
    if [ "$inference_mode" = "--auto_latest_checkpoint" ] && ! compgen -G "${src_exp}/checkpoint-*" > /dev/null; then
        echo "[ERROR] no checkpoint-* found under: $src_exp"
        exit 1
    fi
    if [ "$inference_mode" = "--auto_best_checkpoint" ] && [ ! -d "${src_exp}/checkpoint-best" ]; then
        echo "[ERROR] checkpoint-best not found under: $src_exp"
        exit 1
    fi
    mkdir -p "$analysis_root"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Run greedy and forced semantic STOP decoding"
    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl
        out_root=${src_exp}/${test_set}/${out_subdir}
        records=${out_root}/records.jsonl
        if [ -f "$records" ] && [ "$overwrite" != "true" ]; then
            echo "[skip] existing records: $records (use --overwrite true to rerun)"
            continue
        fi
        mkdir -p "$out_root"
        limit_opt=()
        if [ "$limit" -gt 0 ]; then
            limit_opt=(--limit "$limit")
        fi
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test_semantic_stop.py \
                $inference_mode \
                --exp_dir "$src_exp" \
                --input_jsonl "$test_jsonl" \
                --output_dir "$out_root" \
                --split "$test_set" \
                --device "$device" \
                --decoding_conf "$decoding_conf" \
                --smoke_print "$smoke_print" \
                "${limit_opt[@]}"
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Summarize and plot semantic STOP diagnostics"
    python local/analyze_macslu_semantic_stop.py \
        --root "$src_exp" \
        --records_subdir "$out_subdir" \
        --output_root "$analysis_root" \
        --splits $test_sets
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Evaluate semantic STOP predictions"
    for test_set in $test_sets; do
        gt_file=${json_root}/${test_set}.jsonl
        out_root=${src_exp}/${test_set}/${out_subdir}
        pred_file=${out_root}/predictions.jsonl
        forced_pred_file=${out_root}/forced_predictions.jsonl

        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi

        mkdir -p "${out_root}/metrics_greedy"
        python local/metrics.py \
            --output_dir "${out_root}/metrics_greedy" \
            "$pred_file" "$gt_file" | tee "${out_root}/metrics.txt"

        if [ -f "$forced_pred_file" ]; then
            mkdir -p "${out_root}/metrics_forced"
            python local/metrics.py \
                --output_dir "${out_root}/metrics_forced" \
                "$forced_pred_file" "$gt_file" | tee "${out_root}/forced_metrics.txt"
        fi
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Run visualization for semantic STOP outputs"
    for test_set in $test_sets; do
        out_root=${src_exp}/${test_set}/${out_subdir}
        python local/visualization.py --result_root "$out_root"
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Key statistics and output locations"
    echo "Outputs:"
    for test_set in $test_sets; do
        echo "  ${src_exp}/${test_set}/${out_subdir}/records.jsonl"
        echo "  ${src_exp}/${test_set}/${out_subdir}/predictions.jsonl"
        echo "  ${src_exp}/${test_set}/${out_subdir}/forced_predictions.jsonl"
        echo "  ${src_exp}/${test_set}/${out_subdir}/metrics.txt"
        echo "  ${src_exp}/${test_set}/${out_subdir}/forced_metrics.txt"
    done
    echo "  ${analysis_root}/summary.csv"
    echo "  ${analysis_root}/fig_stop_logprob.{pdf,png}"
    echo "  ${analysis_root}/fig_forced_outcome.{pdf,png}"
fi
