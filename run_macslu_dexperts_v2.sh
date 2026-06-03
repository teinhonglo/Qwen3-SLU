#!/bin/bash
# Prototype-guided MAC-SLU second-pass decoding. No expert LM is trained.

set -euo pipefail

# data / experiment config
json_root=data-json/macslu
labels_path=data/macslu/labels.txt
prototype_root=data-json/macslu_dexperts_v2
exp_root=exp/macslu
inference_mode="--auto_latest_checkpoint"

# model / decoding config
gpuid=0
suffix=
train_conf=conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead.json
decoding_conf=conf/decoding/basic_decoding.json
prototype_source=text_prefix
prototype_pooling=mean_pooling # mean_pooling | last_hidden_state
max_examples_per_label=0 # prototypes use all train examples by default
baseline_mode=reuse  # reuse | run | skip

# replacement config: conservative defaults
domain_threshold=0.35
intent_threshold=0.35
slot_key_threshold=0.35
replacement_margin=0.05
prototype_top_k=5

# t-SNE visualization config
tsne_test_set=test
tsne_perplexity=30
tsne_max_train_examples_per_label=200
tsne_max_test_examples_per_label=200
tsne_random_state=66

# stage config
stage=0
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi
if [ ! -f "$labels_path" ]; then
    echo "[ERROR] labels_path not found: $labels_path"
    exit 1
fi
if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding_conf not found: $decoding_conf"
    exit 1
fi

conf_tag=$(basename -s .json "$train_conf")
base_exp_dir=${exp_root}/${conf_tag}${suffix}
v2_exp_root=${base_exp_dir}/dexperts_v2
schema_path=${prototype_root}/schema.json
prototype_json=${prototype_root}/prototypes_${prototype_source}_${prototype_pooling}.json
prototype_train_examples_jsonl=${prototype_root}/prototype_train_examples_${prototype_source}_${prototype_pooling}.jsonl
prototype_test_examples_jsonl=${prototype_root}/prototype_${tsne_test_set}_examples_${prototype_source}_${prototype_pooling}.jsonl
prototype_fig_dir=${v2_exp_root}/figs

prediction_subdir() {
    local set_name=$1
    local conf_path=$2
    python - "$set_name" "$conf_path" <<'PY'
import json, os, sys
set_name, conf_path = sys.argv[1], sys.argv[2]
mode = "basic"
try:
    with open(conf_path, "r", encoding="utf-8") as f:
        mode = (json.load(f).get("decoding", {}) or {}).get("mode", "basic")
except Exception:
    mode = "basic"
if mode == "basic":
    print(set_name)
else:
    print(f"{set_name}_{os.path.splitext(os.path.basename(conf_path))[0]}")
PY
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Verify MAC-SLU jsonl and labels exist"
    for split in train dev; do
        f=${json_root}/${split}.jsonl
        if [ ! -f "$f" ]; then
            echo "[ERROR] missing required file: $f"
            echo "[HINT] Please run run_macslu.sh stage 0 first."
            exit 1
        fi
    done
    for test_set in $test_sets; do
        f=${json_root}/${test_set}.jsonl
        if [ ! -f "$f" ]; then
            echo "[ERROR] missing required file: $f"
            exit 1
        fi
    done
    f=${json_root}/${tsne_test_set}.jsonl
    if [ ! -f "$f" ]; then
        echo "[ERROR] missing required t-SNE test file: $f"
        exit 1
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Build train/dev schema for prototype filtering"
    mkdir -p "$prototype_root"
    python local/build_macslu_schema.py \
        --input_jsonls ${json_root}/train.jsonl ${json_root}/dev.jsonl \
        --output_json "$schema_path"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Build MAC-SLU prototypes (${prototype_source}; no expert LM training)"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python local/build_macslu_prototypes.py \
            $inference_mode \
            --exp_dir "$base_exp_dir" \
            --train_jsonl ${json_root}/train.jsonl \
            --test_jsonl ${json_root}/${tsne_test_set}.jsonl \
            --labels_path "$labels_path" \
            --schema_path "$schema_path" \
            --output_json "$prototype_json" \
            --train_examples_jsonl "$prototype_train_examples_jsonl" \
            --test_examples_jsonl "$prototype_test_examples_jsonl" \
            --device cuda:0 \
            --prototype_source "$prototype_source" \
            --prototype_pooling "$prototype_pooling" \
            --max_examples_per_label "$max_examples_per_label" \
            --max_instance_examples_per_label "$tsne_max_train_examples_per_label"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Baseline inference mode=${baseline_mode}"
    if [ "$baseline_mode" = "run" ]; then
        for test_set in $test_sets; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python finetuning/qwen3_asr_test.py \
                    $inference_mode \
                    --exp_dir "$base_exp_dir" \
                    --input_jsonl ${json_root}/${test_set}.jsonl \
                    --output_root "$base_exp_dir" \
                    --device cuda:0 \
                    --decoding_conf "$decoding_conf"
        done
    elif [ "$baseline_mode" = "reuse" ]; then
        for test_set in $test_sets; do
            subdir=$(prediction_subdir "$test_set" "$decoding_conf")
            pred_file=${base_exp_dir}/${subdir}/predictions.jsonl
            if [ ! -f "$pred_file" ]; then
                echo "[ERROR] baseline prediction not found: $pred_file"
                echo "[HINT] Use --baseline_mode run or run run_macslu.sh stage 2 first."
                exit 1
            fi
            echo "[info] reuse baseline prediction: $pred_file"
        done
    elif [ "$baseline_mode" = "skip" ]; then
        echo "[info] skip baseline inference/reuse check"
    else
        echo "[ERROR] unsupported baseline_mode: $baseline_mode"
        exit 1
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Prototype-guided second-pass inference"
    for test_set in $test_sets; do
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test_prototype.py \
                $inference_mode \
                --exp_dir "$base_exp_dir" \
                --input_jsonl ${json_root}/${test_set}.jsonl \
                --output_root "$v2_exp_root" \
                --device cuda:0 \
                --decoding_conf "$decoding_conf" \
                --prototype_json "$prototype_json" \
                --labels_path "$labels_path" \
                --schema_path "$schema_path" \
                --prototype_top_k "$prototype_top_k" \
                --domain_threshold "$domain_threshold" \
                --intent_threshold "$intent_threshold" \
                --slot_key_threshold "$slot_key_threshold" \
                --replacement_margin "$replacement_margin"
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Evaluate MAC-SLU predictions (DExperts v2 prototype)"
    for test_set in $test_sets; do
        subdir=$(prediction_subdir "$test_set" "$decoding_conf")
        pred_file=${v2_exp_root}/${subdir}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi
        python local/metrics.py --output_dir ${v2_exp_root}/${subdir} "$pred_file" "$gt_file" | tee ${v2_exp_root}/${subdir}/metrics.txt
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Summary (MAC-SLU DExperts v2 prototype)"
    for test_set in $test_sets; do
        subdir=$(prediction_subdir "$test_set" "$decoding_conf")
        metrics_file=${v2_exp_root}/${subdir}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi
        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi


if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Plot train/test/prototype t-SNE visualizations"
    if [ ! -f "$prototype_json" ]; then
        echo "[ERROR] prototype json not found: $prototype_json"
        exit 1
    fi
    if [ ! -f "$prototype_train_examples_jsonl" ]; then
        echo "[ERROR] train prototype instance jsonl not found: $prototype_train_examples_jsonl"
        exit 1
    fi
    if [ ! -f "$prototype_test_examples_jsonl" ]; then
        echo "[ERROR] test prototype instance jsonl not found: $prototype_test_examples_jsonl"
        exit 1
    fi
    python local/plot_macslu_prototype_tsne.py \
        --prototype_json "$prototype_json" \
        --train_examples_jsonl "$prototype_train_examples_jsonl" \
        --test_examples_jsonl "$prototype_test_examples_jsonl" \
        --output_dir "$prototype_fig_dir" \
        --perplexity "$tsne_perplexity" \
        --max_train_examples_per_label "$tsne_max_train_examples_per_label" \
        --max_test_examples_per_label "$tsne_max_test_examples_per_label" \
        --random_state "$tsne_random_state"
fi
