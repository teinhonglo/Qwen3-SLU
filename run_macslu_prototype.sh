#!/bin/bash
# MAC-SLU prototype bootstrapping pipeline.
#
# Stage 0 prepares the original MAC-SLU JSONL files.
# Stage 1 trains a prototype-only Qwen3-ASR model with full finetuning, uses it
#         to predict train/dev/test domain-intent candidates, writes
#         metrics_proto.txt for every split, and creates ${json_root}_prototype.
# Stage 2 trains the regular MAC-SLU model on the prototype-augmented JSONL data
#         by invoking run_macslu.sh with --json_root.

set -euo pipefail

# data / experiment config
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
exp_root="exp/macslu_fixed"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
json_root="data-json/macslu_fixed"
labels_path=${data_root}/labels.txt
prompt_file=""   # Empty uses prepare_macslu_jsonl.py built-in prompt.

# prototype-only full-finetune config
prototype_train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead_prototype.json"
prototype_top_k=5
checkpoint_mode="best"  # best | latest | exp_dir
skip_prototype_train=0

# downstream MAC-SLU config; run_macslu.sh appends the train-conf tag under this root.
downstream_train_conf="conf/macslu_qwen3_asr_06b.json"
downstream_extra_opts=""

# model/runtime config
gpuid=0
suffix=
seed=66
checkpoint=

# stage config
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

prototype_exp_dir=${exp_root}/prototype
prototype_json_root=${json_root}_prototype
downstream_exp_root=${exp_root}_prototype
prototype_schema_path=${prototype_json_root}/schema.json
prototype_runtime_conf=${prototype_json_root}/prototype_full_runtime.json

if [ ! -f "$prototype_train_conf" ]; then
    echo "[ERROR] prototype_train_conf not found: $prototype_train_conf"
    exit 1
fi
if [ ! -f "$downstream_train_conf" ]; then
    echo "[ERROR] downstream_train_conf not found: $downstream_train_conf"
    exit 1
fi

if [ "$checkpoint" != "" ]; then
    prototype_resume_opts="--resume_from $checkpoint --resume 1"
else
    prototype_resume_opts=""
fi
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download MAC-SLU and prepare original jsonl"
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
    echo "Stage 1: Prototype-only full finetuning, train/dev/test inference, and jsonl generation"
    mkdir -p "$prototype_json_root"

    python local/build_macslu_schema.py \
        --input_jsonls "${json_root}/train.jsonl" "${json_root}/dev.jsonl" \
        --output_json "$prototype_schema_path"

    python - "$prototype_train_conf" "$prototype_runtime_conf" "$labels_path" "$prototype_schema_path" "$prototype_top_k" <<'PY'
import json
import sys
src, dst, labels_path, schema_path, top_k = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
if not isinstance(cfg, list) or len(cfg) != 2:
    raise ValueError("prototype_train_conf must be [training_args, model_args]")
model_args = cfg[1]
model_args.pop("lora_config", None)
model_args["lora_type"] = "full"
proto = dict(model_args.get("prototype", {}) or {})
proto["enabled"] = True
proto["labels_path"] = labels_path
proto["schema_path"] = schema_path
proto["prototype_json"] = ""
proto.pop("init_path", None)
proto["k"] = int(top_k)
model_args["prototype"] = proto
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
    f.write("\n")
print(f"[info] wrote prototype-only runtime config: {dst}")
PY

    if [ "$skip_prototype_train" != "1" ]; then
        CUDA_VISIBLE_DEVICES=$gpuid \
            python finetuning/qwen3_asr_sft_prototype.py \
                --seed "$seed" \
                --train_conf "$prototype_runtime_conf" \
                --train_file "${json_root}/train.jsonl" \
                --eval_file "${json_root}/dev.jsonl" \
                --output_dir "$prototype_exp_dir" \
                --device cuda:0 \
                $prototype_resume_opts
    else
        echo "[info] skip prototype-only training; reuse $prototype_exp_dir"
    fi

    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_test_prototype.py \
            --exp_dir "$prototype_exp_dir" \
            --train_file "${json_root}/train.jsonl" \
            --eval_file "${json_root}/dev.jsonl" \
            --test_file "${json_root}/test.jsonl" \
            --output_jsonl_dir "$prototype_json_root" \
            --prediction_root "$prototype_exp_dir" \
            --prototype_top_k "$prototype_top_k" \
            --checkpoint_mode "$checkpoint_mode" \
            --device cuda:0
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Train MAC-SLU with prototype-augmented jsonl"
    ./run_macslu.sh \
        --json_root "$prototype_json_root" \
        --exp_root "$downstream_exp_root" \
        --stage 1 \
        --train_conf "$downstream_train_conf" \
        --seed "$seed" \
        --gpuid "$gpuid" \
        --suffix "$suffix" \
        $downstream_extra_opts
fi
