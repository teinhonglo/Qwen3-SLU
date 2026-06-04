#!/bin/bash
# End-to-end MAC-SLU pipeline for Qwen3-ASR domain/intent prototype finetuning.
#
# This script intentionally reuses the same PrototypeIndex artifact consumed by
# finetuning/qwen3_asr_test_prototype.py --prototype_json.  Build it with
# local/build_macslu_prototypes.py, then pass that JSON to
# finetuning/qwen3_asr_sft_prototype.py through a generated train config.

set -euo pipefail

# data config
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
exp_root="exp/macslu"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
json_root=data-json/macslu
labels_path=${data_root}/labels.txt
prompt_file=""   # Same behavior as run_macslu.sh: empty uses prepare_macslu_jsonl.py built-in prompt.

# model / decoding config
nj=4
gpuid=0
suffix=
seed=66
inference_mode="--auto_latest_checkpoint"
decoding_conf="conf/decoding/basic_decoding.json"
checkpoint=

# Prototype SFT config.  This config contains the prototype section; the script
# writes a runtime copy whose prototype.prototype_json points at the artifact
# produced in Stage 3 below.
train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead_prototype.json"

# Prototype extraction config.  local/build_macslu_prototypes.py needs an
# existing Qwen3-ASR experiment directory because it extracts hidden-state
# prototypes from a trained/non-prototype ASR-SLU model.  By default we use the
# matching non-prototype config/experiment from run_macslu.sh.
base_train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead.json"
base_exp_dir=""
run_base_train=0       # 0: require/reuse base_exp_dir, 1: train it in Stage 1.
base_checkpoint=

# Prototype artifact config.  prototype_json is the exact file used to initialize
# qwen3_asr_sft_prototype.py and is compatible with qwen3_asr_test_prototype.py.
prototype_root=data-json/macslu_prototype
prototype_source=text_prefix       # text_prefix | audio_prefix
prototype_pooling=mean_pooling     # mean_pooling | last_hidden_state
prototype_json=""
prototype_train_examples_jsonl=""
prototype_test_examples_jsonl=""
max_examples_per_label=0
max_instance_examples_per_label=200
tsne_test_set=test

# Inference config for the prototype-finetuned model.
prototype_top_k=5
attention_map_opts="" # reserved for parity with run_macslu.sh; prototype test script currently ignores it.

# stage config
stage=0
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] prototype train_conf not found: $train_conf"
    exit 1
fi
if [ ! -f "$base_train_conf" ]; then
    echo "[ERROR] base_train_conf not found: $base_train_conf"
    exit 1
fi
if [ ! -f "$decoding_conf" ]; then
    echo "[ERROR] decoding_conf not found: $decoding_conf"
    exit 1
fi

conf_tag=$(basename -s .json "$train_conf")
base_conf_tag=$(basename -s .json "$base_train_conf")
exp_dir=${exp_root}/${conf_tag}${suffix}
if [ -z "$base_exp_dir" ]; then
    base_exp_dir=${exp_root}/${base_conf_tag}${suffix}
fi

schema_path=${prototype_root}/schema.json
if [ -z "$prototype_json" ]; then
    prototype_json=${prototype_root}/prototypes_${prototype_source}_${prototype_pooling}.json
fi
if [ -z "$prototype_train_examples_jsonl" ]; then
    prototype_train_examples_jsonl=${prototype_root}/prototype_train_examples_${prototype_source}_${prototype_pooling}.jsonl
fi
if [ -z "$prototype_test_examples_jsonl" ]; then
    prototype_test_examples_jsonl=${prototype_root}/prototype_${tsne_test_set}_examples_${prototype_source}_${prototype_pooling}.jsonl
fi
runtime_train_conf=${prototype_root}/${conf_tag}_runtime.json

if [ "$checkpoint" != "" ]; then
    training_opts="--resume_from $checkpoint --resume 1"
else
    training_opts=""
fi
if [ "$base_checkpoint" != "" ]; then
    base_training_opts="--resume_from $base_checkpoint --resume 1"
else
    base_training_opts=""
fi

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
    echo "Stage 0: Download MAC-SLU and prepare jsonl"
    prep_cmd=(
        python local/prepare_macslu_jsonl.py
        --repo-id "$repo_id"
        --download-dir "$download_dir"
        --extract-root "$extract_root"
        --jsonl-root "$json_root"
        --splits train dev test
    )
    if [ -n "$prompt_file" ]; then
        prep_cmd+=(--prompt-file "$prompt_file")
    fi
    "${prep_cmd[@]}"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Prepare/reuse base Qwen3-ASR experiment for prototype extraction"
    if [ "$run_base_train" = "1" ]; then
        CUDA_VISIBLE_DEVICES=$gpuid \
            python finetuning/qwen3_asr_sft.py --seed $seed $base_training_opts \
                --train_conf "$base_train_conf" \
                --train_file ${json_root}/train.jsonl \
                --eval_file ${json_root}/dev.jsonl \
                --output_dir "$base_exp_dir"
    else
        if [ ! -f "$base_exp_dir/train_conf.json" ]; then
            echo "[ERROR] base experiment not found: $base_exp_dir/train_conf.json"
            echo "[HINT] Run run_macslu.sh with --train_conf $base_train_conf first, or rerun this script with --run_base_train 1."
            exit 1
        fi
        echo "[info] reuse base experiment: $base_exp_dir"
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Build train/dev schema for prototype filtering"
    mkdir -p "$prototype_root"
    python local/build_macslu_schema.py \
        --input_jsonls ${json_root}/train.jsonl ${json_root}/dev.jsonl \
        --output_json "$schema_path"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Build PrototypeIndex JSON for domain/intent initialization"
    mkdir -p "$prototype_root"
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
            --max_instance_examples_per_label "$max_instance_examples_per_label"
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Write runtime prototype train config"
    if [ ! -f "$prototype_json" ]; then
        echo "[ERROR] prototype_json not found: $prototype_json"
        echo "[HINT] Run Stage 3 first."
        exit 1
    fi
    mkdir -p "$prototype_root"
    python - "$train_conf" "$runtime_train_conf" "$prototype_json" "$labels_path" "$schema_path" "$prototype_top_k" <<'PY'
import json, sys
src, dst, prototype_json, labels_path, schema_path, prototype_top_k = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
if not isinstance(cfg, list) or len(cfg) != 2:
    raise ValueError("train_conf must be [training_args, model_args]")
model_args = cfg[1]
proto = dict(model_args.get("prototype", {}) or {})
proto["enabled"] = True
proto["prototype_json"] = prototype_json
proto.pop("init_path", None)
proto["labels_path"] = labels_path
proto["schema_path"] = schema_path
proto["k"] = int(prototype_top_k)
model_args["prototype"] = proto
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
    f.write("\n")
print(f"[info] wrote runtime train config: {dst}")
print(f"[info] prototype_json: {prototype_json}")
PY
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Prototype-aware finetuning on MAC-SLU"
    if [ ! -f "$runtime_train_conf" ]; then
        echo "[ERROR] runtime_train_conf not found: $runtime_train_conf"
        echo "[HINT] Run Stage 4 first."
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_sft_prototype.py --seed $seed $training_opts \
            --train_conf "$runtime_train_conf" \
            --train_file ${json_root}/train.jsonl \
            --eval_file ${json_root}/dev.jsonl \
            --output_dir "$exp_dir"
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Prototype-aware inference on MAC-SLU test"
    for test_set in $test_sets; do
        test_jsonl=${json_root}/${test_set}.jsonl
        mkdir -p ${exp_dir}/${test_set}
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python finetuning/qwen3_asr_test_domain_intent_prototype.py \
                $inference_mode \
                --exp_dir "$exp_dir" \
                --input_jsonl "$test_jsonl" \
                --output_root "$exp_dir" \
                --device cuda:0 \
                --decoding_conf "$decoding_conf" \
                --prototype_top_k "$prototype_top_k"
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Evaluate MAC-SLU prototype predictions"
    for test_set in $test_sets; do
        subdir=$(prediction_subdir "$test_set" "$decoding_conf")
        pred_file=${exp_dir}/${subdir}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi
        python local/metrics.py --output_dir ${exp_dir}/${subdir} "$pred_file" "$gt_file" | tee ${exp_dir}/${subdir}/metrics.txt
    done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "Stage 8: Plot MAC-SLU prototype confusion matrices"
    for test_set in $test_sets; do
        subdir=$(prediction_subdir "$test_set" "$decoding_conf")
        pred_file=${exp_dir}/${subdir}/predictions.jsonl
        gt_file=${json_root}/${test_set}.jsonl
        output_dir=${exp_dir}/${subdir}
        if [ ! -f "$pred_file" ]; then
            echo "[WARNING] prediction file not found: $pred_file"
            continue
        fi
        if [ ! -f "$gt_file" ]; then
            echo "[WARNING] ground truth file not found: $gt_file"
            continue
        fi
        python local/plot_macslu_confusion.py \
            --pred_file "$pred_file" \
            --gt_file "$gt_file" \
            --labels_file "${data_root}/labels.txt" \
            --label_mapping_file "${data_root}/labels_zh_en.txt" \
            --output_dir "$output_dir"
    done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    echo "Stage 9: Summary (MAC-SLU prototype)"
    for test_set in $test_sets; do
        subdir=$(prediction_subdir "$test_set" "$decoding_conf")
        metrics_file=${exp_dir}/${subdir}/metrics.txt
        if [ ! -f "$metrics_file" ]; then
            echo "[WARNING] metrics file not found: $metrics_file"
            continue
        fi
        echo "========== ${test_set} =========="
        cat "$metrics_file"
    done
fi
