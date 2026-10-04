#!/bin/bash
# Decode and evaluate every retained checkpoint-N directory of one MAC-SLU
# experiment. Use eval_set=test to report test metrics. Use eval_set=dev to
# select the checkpoint with the highest validation semantic-frame EMA.

set -euo pipefail

json_root="data-json/macslu_fixed"
exp_dir=""
eval_set="test"  # test: report only; dev: also write selected_checkpoint.txt.
decoding_conf="conf/decoding/basic_decoding.json"
gpuid=0
seed=66
output_root=""  # Empty writes under <exp_dir>/checkpoint_metrics.
force=0         # 1 reruns inference even when predictions already exist.

. ./local/parse_options.sh
. ./path.sh

if [ -z "$exp_dir" ]; then
    echo "[ERROR] --exp_dir is required"
    exit 1
fi

if [ ! -d "$exp_dir" ]; then
    echo "[ERROR] experiment directory not found: $exp_dir"
    exit 1
fi

if [ ! -f "${exp_dir}/train_conf.json" ]; then
    echo "[ERROR] train_conf.json not found under: $exp_dir"
    exit 1
fi

input_jsonl="${json_root}/${eval_set}.jsonl"
if [ ! -f "$input_jsonl" ]; then
    echo "[ERROR] evaluation jsonl not found: $input_jsonl"
    exit 1
fi

if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding config not found: $decoding_conf"
    exit 1
fi

decoding_conf_name=$(basename -s .json "$decoding_conf")
if [ -z "$output_root" ]; then
    output_root="${exp_dir}/checkpoint_metrics"
fi
run_output_root="${output_root}/${eval_set}_${decoding_conf_name}"
mkdir -p "$run_output_root"

mapfile -t checkpoint_names < <(
    find "$exp_dir" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
        | awk '/^checkpoint-[0-9]+$/' \
        | sort -V
)

if [ "${#checkpoint_names[@]}" -eq 0 ]; then
    echo "[ERROR] no checkpoint-N directories found under: $exp_dir"
    exit 1
fi

summary_file="${run_output_root}/metrics_summary.tsv"
summary_tmp="${summary_file}.tmp"
printf 'checkpoint\toverall_accuracy\tintent_accuracy\tslot_f1\texplicit_slot_f1\timplicit_slot_f1\tquery_mer\tmetrics_file\n' > "$summary_tmp"

for checkpoint_name in "${checkpoint_names[@]}"; do
    checkpoint_path="${exp_dir}/${checkpoint_name}"
    checkpoint_output_root="${run_output_root}/${checkpoint_name}"
    result_dir="${checkpoint_output_root}/${eval_set}_${decoding_conf_name}"
    prediction_file="${result_dir}/predictions.jsonl"
    metrics_file="${result_dir}/metrics.txt"

    echo "========== ${checkpoint_name} =========="
    if [ "$force" = "1" ] || [ ! -f "$prediction_file" ]; then
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test.py \
                --exp_dir "$exp_dir" \
                --checkpoint "$checkpoint_path" \
                --input_jsonl "$input_jsonl" \
                --output_root "$checkpoint_output_root" \
                --device cuda:0 \
                --decoding_conf "$decoding_conf" \
                --seed "$seed"
    else
        echo "[info] reuse existing predictions: $prediction_file"
    fi

    python local/metrics.py \
        --output_dir "$result_dir" \
        "$prediction_file" \
        "$input_jsonl" \
        | tee "$metrics_file"

    overall_accuracy=$(awk '/^Overall accuracy:/ {print $3; exit}' "$metrics_file")
    intent_accuracy=$(awk '/^Intent accuracy:/ {print $3; exit}' "$metrics_file")
    slot_f1=$(awk '/^Slot P\/R\/F1:/ {print $NF; exit}' "$metrics_file")
    explicit_slot_f1=$(awk '/^Explicit Slot P\/R\/F1:/ {print $NF; exit}' "$metrics_file")
    implicit_slot_f1=$(awk '/^Implicit Slot P\/R\/F1:/ {print $NF; exit}' "$metrics_file")
    query_mer=$(awk '/^Query MER:/ {print $3; exit}' "$metrics_file")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$checkpoint_name" \
        "$overall_accuracy" \
        "$intent_accuracy" \
        "$slot_f1" \
        "$explicit_slot_f1" \
        "$implicit_slot_f1" \
        "$query_mer" \
        "$metrics_file" \
        >> "$summary_tmp"
done

{
    head -n 1 "$summary_tmp"
    tail -n +2 "$summary_tmp" | sort -t $'\t' -k2,2gr
} > "$summary_file"
rm "$summary_tmp"

echo "[info] checkpoint metric summary: $summary_file"

if [ "$eval_set" = "dev" ]; then
    selected_checkpoint=$(awk -F '\t' 'NR == 2 {print $1}' "$summary_file")
    selected_checkpoint_path="${exp_dir}/${selected_checkpoint}"
    printf '%s\n' "$selected_checkpoint_path" > "${run_output_root}/selected_checkpoint.txt"
    echo "[info] selected by validation EMA: $selected_checkpoint_path"
    echo "[info] selection record: ${run_output_root}/selected_checkpoint.txt"
else
    echo "[info] ${eval_set} metrics are reported only and are not used for checkpoint selection"
fi
