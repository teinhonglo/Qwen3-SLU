#!/bin/bash
# MAC-SLU prototype bootstrapping pipeline.
#
# Stage 0 builds the MAC-SLU schema used by prototype label maps.
# Stage 1 trains a seed prototype-only Qwen3-ASR model with random prototype
#         initialization.
# Stage 2 extracts hidden-state domain/intent prototypes from the seed model by
#         invoking local/build_macslu_prototypes.py.
# Stage 3 trains the final prototype-only Qwen3-ASR model initialized from the
#         extracted prototype JSON.
# Stage 4 uses the final prototype-only model to predict train/dev/test
#         domain-intent candidates, writes metrics_proto.txt for every split,
#         and creates ${json_root}_prototype.
# Stage 5 trains the regular MAC-SLU model on the prototype-augmented JSONL data
#         by invoking run_macslu.sh with --json_root.

set -euo pipefail

# data / experiment config

data_root="data/macslu"
exp_root="exp/macslu_fixed"
json_root="data-json/macslu_fixed"
labels_path=${data_root}/labels.txt
prompt_file=""   # Empty uses prepare_macslu_jsonl.py built-in prompt.

# prototype-only full-finetune config
prototype_train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead_prototype.json"

prototype_top_k=5
prototype_source="audio_only"       # Match local/build_macslu_prototypes.py default: audio_only | audio_prompt | audio_prefix | text_prefix
prototype_pooling="mean_pooling"     # Match original prototype extraction default: mean_pooling | last_hidden_state

checkpoint_mode="best"  # best | latest | exp_dir
prototype_build_checkpoint_mode="best"  # best | latest | exp_dir; checkpoint used by local/build_macslu_prototypes.py
skip_seed_prototype_train=0
skip_build_prototypes=0
skip_prototype_train=0       # Skip final prototype-only training; reuse $prototype_exp_dir.

# downstream MAC-SLU config; run_macslu.sh appends the train-conf tag under this root.
downstream_train_conf="conf/macslu_qwen3_asr_17b_ep10_lora_woemblmhead.json"
downstream_exp_root="exp/macslu_prototype"
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

prototype_json_root=${json_root}_prototype
downstream_exp_root=${exp_root}_prototype
prototype_schema_path=${prototype_json_root}/schema.json
prototype_seed_exp_dir=${exp_root}/prototype_seed
prototype_exp_dir=${exp_root}/prototype
prototype_seed_runtime_conf=${prototype_json_root}/prototype_seed_runtime.json
prototype_runtime_conf=${prototype_json_root}/prototype_initialized_runtime.json
prototype_init_json=${prototype_json_root}/prototype_init.json
prototype_train_examples_jsonl=${prototype_json_root}/prototype_train_examples.jsonl
prototype_test_examples_jsonl=${prototype_json_root}/prototype_test_examples.jsonl

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

write_prototype_runtime_conf() {
    local output_conf=$1
    local init_json=$2
    python - "$prototype_train_conf" "$output_conf" "$labels_path" "$prototype_schema_path" "$prototype_top_k" "$prototype_source" "$prototype_pooling" "$init_json" <<'PY'
import json
import sys

src, dst, labels_path, schema_path, top_k, prototype_source, prototype_pooling, init_json = sys.argv[1:]
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
if init_json:
    proto["prototype_json"] = init_json
else:
    proto["prototype_json"] = ""
proto.pop("init_path", None)
proto["k"] = int(top_k)
proto["prototype_source"] = prototype_source
proto["pooling"] = prototype_pooling
model_args["prototype"] = proto
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
    f.write("\n")
print(f"[info] wrote prototype runtime config: {dst}; init_json={init_json or '<random>'}")
PY
}

prototype_checkpoint_opt() {
    local mode=$1
    if [ "$mode" = "best" ]; then
        echo "--auto_best_checkpoint"
    elif [ "$mode" = "latest" ]; then
        echo "--auto_latest_checkpoint"
    elif [ "$mode" = "exp_dir" ]; then
        echo ""
    else
        echo "[ERROR] unsupported checkpoint mode: $mode" >&2
        return 1
    fi
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Build MAC-SLU prototype schema"
    mkdir -p "$prototype_json_root"

    python local/build_macslu_schema.py \
        --input_jsonls "${json_root}/train.jsonl" "${json_root}/dev.jsonl" \
        --output_json "$prototype_schema_path"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Train seed prototype-only model with random initialization"
    mkdir -p "$prototype_json_root"
    if [ ! -f "$prototype_schema_path" ]; then
        python local/build_macslu_schema.py \
            --input_jsonls "${json_root}/train.jsonl" "${json_root}/dev.jsonl" \
            --output_json "$prototype_schema_path"
    fi
    write_prototype_runtime_conf "$prototype_seed_runtime_conf" ""

    if [ "$skip_seed_prototype_train" != "1" ]; then
        CUDA_VISIBLE_DEVICES=$gpuid \
            python finetuning/qwen3_asr_sft_prototype.py \
                --seed "$seed" \
                --train_conf "$prototype_seed_runtime_conf" \
                --train_file "${json_root}/train.jsonl" \
                --eval_file "${json_root}/dev.jsonl" \
                --output_dir "$prototype_seed_exp_dir" \
                --device cuda:0 \
                $prototype_resume_opts
    else
        echo "[info] skip seed prototype-only training; reuse $prototype_seed_exp_dir"
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Build prototype initialization JSON from seed model"
    mkdir -p "$prototype_json_root"
    if [ "$skip_build_prototypes" != "1" ]; then
        build_checkpoint_opt=$(prototype_checkpoint_opt "$prototype_build_checkpoint_mode")
        CUDA_VISIBLE_DEVICES=$gpuid \
            python local/build_macslu_prototypes.py \
                --train_jsonl "${json_root}/train.jsonl" \
                --test_jsonl "${json_root}/test.jsonl" \
                --labels_path "$labels_path" \
                --schema_path "$prototype_schema_path" \
                --output_json "$prototype_init_json" \
                --train_examples_jsonl "$prototype_train_examples_jsonl" \
                --test_examples_jsonl "$prototype_test_examples_jsonl" \
                --exp_dir "$prototype_seed_exp_dir" \
                $build_checkpoint_opt \
                --device cuda:0 \
                --prototype_source "$prototype_source" \
                --prototype_pooling "$prototype_pooling"
    else
        echo "[info] skip prototype JSON build; reuse $prototype_init_json"
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Train final prototype-only model initialized from prototype JSON"
    mkdir -p "$prototype_json_root"
    if [ ! -f "$prototype_init_json" ]; then
        echo "[ERROR] prototype initialization JSON not found: $prototype_init_json"
        echo "        Run stage 2 first or set --skip_build_prototypes 0."
        exit 1
    fi
    write_prototype_runtime_conf "$prototype_runtime_conf" "$prototype_init_json"

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
        echo "[info] skip final prototype-only training; reuse $prototype_exp_dir"
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Prototype train/dev/test inference and jsonl generation"
    mkdir -p "$prototype_json_root"
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

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Train MAC-SLU with prototype-augmented jsonl"
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
