#!/bin/bash
# MAC-SLU prototype bootstrapping pipeline.
#
# Stage 0 builds the MAC-SLU schema used by prototype label maps.

# Stage 1 builds domain/intent prototype vectors from the labels in json_root.
#         If src_model is non-empty, hidden states are extracted from that
#         experiment/checkpoint. If src_model is empty, the model is initialized
#         from downstream_train_conf and used only as the embedding source.
# Stage 2 trains the prototype-only Qwen3-ASR model initialized from the Stage 1
#         prototype JSON.
# Stage 3 uses the trained prototype-only model to predict train/dev/test
#         domain-intent candidates, writes metrics_proto.txt for every split,
#         and creates ${json_root}_prototype.
# Stage 4 trains the regular MAC-SLU model on the prototype-augmented JSONL data
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
prototype_min_similarity="-1"       # -1 auto-selects on dev; empty keeps all top-k candidates in generated data-json prompts.
prototype_metric_ks="1 3 5"       # IR metric cutoffs used by Stage 3.
prototype_source="audio_only"       # Match local/build_macslu_prototypes.py default: audio_only | audio_prompt | audio_prefix | text_prefix
prototype_pooling="mean_pooling"     # Match original prototype extraction default: mean_pooling | last_hidden_state

# Step 1 source model for prototype extraction. Empty means initialize the source
# model from downstream_train_conf instead of loading an existing experiment.
src_model="exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead"
prototype_build_checkpoint_mode="latest"  # best | latest | exp_dir; used only when src_model is non-empty.
checkpoint_mode="best"  # best | latest | exp_dir; checkpoint used by Stage 3 inference.
skip_build_prototypes=0
skip_prototype_train=0
skip_prototype_tsne=0

# downstream MAC-SLU config; run_macslu.sh appends the train-conf tag under this root.
downstream_train_conf="conf/macslu_qwen3_asr_17b_ep20_lora_woemblmhead.json"
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
prototype_exp_dir=${exp_root}/prototype
prototype_runtime_conf=${prototype_json_root}/prototype_runtime.json
prototype_init_json=${prototype_json_root}/prototype_init.json
prototype_train_examples_jsonl=${prototype_json_root}/prototype_train_examples.jsonl
prototype_test_examples_jsonl=${prototype_json_root}/prototype_test_examples.jsonl
prototype_tsne_root=${prototype_json_root}/prototype_tsne

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

proto["prototype_json"] = init_json
proto.pop("init_path", None)
proto["k"] = int(top_k)
proto["prototype_source"] = prototype_source
proto["pooling"] = prototype_pooling
model_args["prototype"] = proto
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
    f.write("\n")
print(f"[info] wrote prototype runtime config: {dst}; init_json={init_json}")
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
    
    python local/count_macslu_intent_distribution.py \
        --jsonl-root "$json_root" \
        --splits train dev test \
        --output-txt "${json_root}/intent_distribution.txt" \
        --output-json "${json_root}/intent_distribution.json"

    python local/build_macslu_schema.py \
        --input_jsonls "${json_root}/train.jsonl" "${json_root}/dev.jsonl" \
        --output_json "$prototype_schema_path"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Build prototype initialization JSON from json_root labels"
    mkdir -p "$prototype_json_root"
    if [ ! -f "$prototype_schema_path" ]; then
        echo "[ERROR] prototype schema not found: $prototype_schema_path"
        echo "        Run stage 0 first or set --prototype_schema_path to an existing schema."
        exit 1
    fi

    if [ "$skip_build_prototypes" != "1" ]; then
        build_source_opts=()
        if [ "$src_model" != "" ]; then
            build_checkpoint_opt=$(prototype_checkpoint_opt "$prototype_build_checkpoint_mode")
            build_source_opts+=(--exp_dir "$src_model")
            if [ "$build_checkpoint_opt" != "" ]; then
                build_source_opts+=("$build_checkpoint_opt")
            fi
            echo "[info] build prototypes from src_model: $src_model"
        else
            build_source_opts+=(--train_conf "$downstream_train_conf")
            echo "[info] src_model is empty; build prototypes from downstream_train_conf: $downstream_train_conf"
        fi
        
        CUDA_VISIBLE_DEVICES=$gpuid \
            python local/build_macslu_prototypes.py \
                --train_jsonl "${json_root}/train.jsonl" \
                --test_jsonl "${json_root}/test.jsonl" \
                --labels_path "$labels_path" \
                --schema_path "$prototype_schema_path" \
                --output_json "$prototype_init_json" \
                --train_examples_jsonl "$prototype_train_examples_jsonl" \
                --test_examples_jsonl "$prototype_test_examples_jsonl" \
                "${build_source_opts[@]}" \
                --device cuda:0 \
                --prototype_source "$prototype_source" \
                --prototype_pooling "$prototype_pooling"
    else
        echo "[info] skip prototype JSON build; reuse $prototype_init_json"
    fi

    if [ "$skip_prototype_tsne" != "1" ]; then
        python local/plot_macslu_prototype_tsne.py \
            --prototype_json "$prototype_init_json" \
            --train_examples_jsonl "$prototype_train_examples_jsonl" \
            --test_examples_jsonl "$prototype_test_examples_jsonl" \
            --output_dir "${prototype_tsne_root}/before_train" \
            --random_state "$seed"
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Train prototype-only model initialized from prototype JSON"
    mkdir -p "$prototype_json_root"
    if [ ! -f "$prototype_init_json" ]; then
        echo "[ERROR] prototype initialization JSON not found: $prototype_init_json"
        echo "        Run stage 1 first or set --prototype_init_json to an existing file."
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

    if [ "$skip_prototype_tsne" != "1" ]; then
        python local/plot_macslu_prototype_tsne.py \
            --prototype_json "$prototype_init_json" \
            --train_examples_jsonl "$prototype_train_examples_jsonl" \
            --test_examples_jsonl "$prototype_test_examples_jsonl" \
            --output_dir "${prototype_tsne_root}/after_train" \
            --random_state "$seed"
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Prototype train/dev/test inference and jsonl generation"
    mkdir -p "$prototype_json_root"
    prototype_infer_opts=(--prototype_metric_ks $prototype_metric_ks)
    if [ -n "$prototype_min_similarity" ]; then
        prototype_infer_opts+=(--prototype_min_similarity "$prototype_min_similarity")
    fi
    CUDA_VISIBLE_DEVICES=$gpuid \
        python finetuning/qwen3_asr_test_prototype.py \
            --exp_dir "$prototype_exp_dir" \
            --train_file "${json_root}/train.jsonl" \
            --dev_file "${json_root}/dev.jsonl" \
            --test_file "${json_root}/test.jsonl" \
            --output_jsonl_dir "$prototype_json_root" \
            --prediction_root "$prototype_exp_dir" \
            --prototype_top_k "$prototype_top_k" \
            "${prototype_infer_opts[@]}" \
            --checkpoint_mode "$checkpoint_mode" \
            --device cuda:0
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Train MAC-SLU with prototype-augmented jsonl"
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
