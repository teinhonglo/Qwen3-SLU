#!/bin/bash
# Shared shell helpers for both controlled MAC-SLU experiments.
resolve_teacher_checkpoint() {
    local exp_dir=$1 mode=$2
    python - "$exp_dir" "$mode" <<'PY'
import sys
sys.path.insert(0, "finetuning")
from qwen3_asr_common import resolve_checkpoint
print(resolve_checkpoint(sys.argv[1], sys.argv[2]))
PY
}

resolve_or_validate_teacher_checkpoint() {
    local explicit_checkpoint=$1 exp_dir=$2 mode=$3
    if [ -n "$explicit_checkpoint" ]; then
        python - "$explicit_checkpoint" <<'PY'
import os, sys
checkpoint = os.path.realpath(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(checkpoint):
    raise SystemExit(f"[ERROR] Explicit teacher checkpoint does not exist: {checkpoint}")
required = ("config.json", "adapter_config.json", "model.safetensors", "pytorch_model.bin")
if not any(os.path.isfile(os.path.join(checkpoint, name)) for name in required):
    raise SystemExit("[ERROR] Explicit teacher checkpoint contains neither a model/config nor a PEFT adapter: " + checkpoint)
print(checkpoint)
PY
    else
        resolve_teacher_checkpoint "$exp_dir" "$mode"
    fi
}

teacher_setting_from_conf() {
    local student_conf=$1 setting=$2
    python - "$student_conf" "$setting" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
teacher = config[1].get("teacher", {})
value = teacher.get(sys.argv[2], "")
print("" if value is None else value)
PY
}

ensure_macslu_jsonl() {
    local root=$1
    if [ ! -f "$root/train.jsonl" ] || [ ! -f "$root/dev.jsonl" ] || [ ! -f "$root/test.jsonl" ]; then
        ./run_macslu.sh --stage 0 --stop_stage 0
    fi
    for split in train dev test; do test -s "$root/$split.jsonl" || { echo "[ERROR] Missing $root/$split.jsonl" >&2; return 1; }; done
}

train_or_verify_teacher() {
    local skip=$1 conf=$2 root=$3 gpu=$4 seed=$5
    if [ "$skip" -eq 0 ]; then
        ./run_macslu.sh --stage 1 --stop_stage 1 --gpuid "$gpu" --seed "$seed" --train_conf "$conf" --exp_root "$root"
    fi
}

train_distillation_teacher() {
    local skip=$1 use_vocabulary_pruning=$2 vocabulary_manifest=$3
    local gpuid=$4 seed=$5 teacher_conf=$6 train_jsonl=$7 dev_jsonl=$8
    local teacher_exp_dir=$9 teacher_source_checkpoint=${10}

    if [ "$skip" -eq 1 ]; then
        echo "[info] Reusing teacher: $teacher_exp_dir"
        return
    fi
    if [ "$use_vocabulary_pruning" = true ]; then
        test -s "$vocabulary_manifest" || {
            echo "[ERROR] Vocabulary manifest not found: $vocabulary_manifest" >&2
            return 1
        }
    fi
    test -d "$teacher_source_checkpoint" || {
        echo "[ERROR] Teacher source checkpoint not found: $teacher_source_checkpoint" >&2
        return 1
    }

    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_sft.py \
            --seed "$seed" \
            --train_conf "$teacher_conf" \
            --train_file "$train_jsonl" \
            --eval_file "$dev_jsonl" \
            --output_dir "$teacher_exp_dir" \
            --init_from_checkpoint "$teacher_source_checkpoint"
}
