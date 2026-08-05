#!/bin/bash
# MAC-SLU PruneSLU-style distillation with scalar K-means weight quantization.

set -euo pipefail

# Stages.
stage=0
stop_stage=9

# Runtime.
gpuid=0
seed=66
resume_checkpoint=""

# Data and vocabulary.  These defaults deliberately match Version A.
json_root="data-json/macslu"
use_vocabulary_pruning=false
vocabulary_model_path="Qwen/Qwen3-ASR-1.7B"
vocabulary_top_frequency_tokens=2000
qwen_frequency_file=""
vocabulary_manifest="data-json/macslu/vocabulary/qwen3_asr_macslu_top2000.json"

# Teacher. Version B defaults to reusing the exact matching teacher produced by
# Version A (full vocabulary or pruned vocabulary, according to the option).
teacher_train_conf="conf/macslu_qwen3_asr_17b_pruneslu_teacher.json"
teacher_source_checkpoint=""
teacher_exp_root="exp/macslu_pruneslu_teacher"
teacher_checkpoint_mode="best"
skip_teacher_train=1

# Student.
student_train_conf="conf/macslu_qwen3_asr_06b_pruneslu_kd_kmeans.json"
student_exp_root="exp/macslu_distillation_kmeans"
distillation_only_exp_dir=""

# Inference/evaluation.
decoding_conf="conf/decoding/basic_decoding.json"
labels_file="data/macslu/labels.txt"
label_mapping_file="data/macslu/labels_zh_en.txt"

. ./local/parse_options.sh
. ./path.sh
. ./local/macslu_distillation_lib.sh

base_teacher_train_conf="$teacher_train_conf"
base_student_train_conf="$student_train_conf"
teacher_tag=$(basename -s .json "$teacher_train_conf")
student_tag=$(basename -s .json "$student_train_conf")

if [ "$use_vocabulary_pruning" = true ]; then
    vocabulary_variant="vocabprune_top${vocabulary_top_frequency_tokens}"
else
    vocabulary_variant="fullvocab"
fi

teacher_exp_dir="$teacher_exp_root/${teacher_tag}_${vocabulary_variant}"
student_exp_dir="$student_exp_root/${student_tag}_${vocabulary_variant}"
teacher_train_conf="$teacher_exp_dir/runtime_train_conf.json"
student_train_conf="$student_exp_dir/runtime_train_conf.json"

conf_teacher_source=$(teacher_setting_from_conf "$base_student_train_conf" teacher_source_checkpoint)
conf_teacher_mode=$(teacher_setting_from_conf "$base_student_train_conf" checkpoint_mode)

[ -n "$teacher_source_checkpoint" ] || teacher_source_checkpoint="$conf_teacher_source"
[ -z "$conf_teacher_mode" ] || teacher_checkpoint_mode="$conf_teacher_mode"

if [ "$use_vocabulary_pruning" = true ]; then
    teacher_for_student="$teacher_exp_dir/checkpoint-best"
else
    teacher_for_student="$teacher_source_checkpoint"
fi

python local/prepare_macslu_distillation_conf.py \
    --input "$base_teacher_train_conf" \
    --output "$teacher_train_conf" \
    --vocabulary_pruning "$use_vocabulary_pruning" \
    --vocabulary_manifest "$vocabulary_manifest"

python local/prepare_macslu_distillation_conf.py \
    --input "$base_student_train_conf" \
    --output "$student_train_conf" \
    --vocabulary_pruning "$use_vocabulary_pruning" \
    --vocabulary_manifest "$vocabulary_manifest" \
    --teacher_exp_dir "$teacher_exp_dir"

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "Stage 0: Prepare or verify MAC-SLU JSONL"
    ensure_macslu_jsonl "$json_root"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    if [ "$use_vocabulary_pruning" = true ]; then
        echo "Stage 1: Compute MAC-SLU/Qwen3-ASR vocabulary"
        vocabulary_frequency_options=()
        if [ -n "$qwen_frequency_file" ]; then
            vocabulary_frequency_options=(--qwen_frequency_file "$qwen_frequency_file")
        fi
        python local/compute_macslu_vocabulary.py \
            --model_path "$vocabulary_model_path" \
            --jsonl \
                "$json_root/train.jsonl" \
                "$json_root/dev.jsonl" \
                "$json_root/test.jsonl" \
            --top_frequency_tokens "$vocabulary_top_frequency_tokens" \
            "${vocabulary_frequency_options[@]}" \
            --output "$vocabulary_manifest"
    else
        echo "Stage 1: Vocabulary pruning disabled; use the full Qwen3-ASR vocabulary"
    fi
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    if [ "$use_vocabulary_pruning" = true ]; then
        echo "Stage 2: Build or reuse vocabulary-pruned teacher with supervised fine-tuning"
        train_distillation_teacher \
            "$skip_teacher_train" \
            "$use_vocabulary_pruning" \
            "$vocabulary_manifest" \
            "$gpuid" \
            "$seed" \
            "$teacher_train_conf" \
            "$json_root/train.jsonl" \
            "$json_root/dev.jsonl" \
            "$teacher_exp_dir" \
            "$teacher_source_checkpoint"
    else
        echo "Stage 2: Vocabulary pruning disabled; use teacher_source_checkpoint directly"
    fi
fi

if [ "$stop_stage" -ge 3 ]; then
    teacher_for_student=$(resolve_or_validate_teacher_checkpoint \
        "$teacher_for_student" \
        "$teacher_exp_dir" \
        "$teacher_checkpoint_mode")
    teacher_exp_dir=$(dirname "$teacher_for_student")
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Stage 3: Validate teacher/student compatibility"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_distillation_kmeans.py \
            --validate_only \
            --seed "$seed" \
            --train_conf "$student_train_conf" \
            --train_file "$json_root/train.jsonl" \
            --eval_file "$json_root/dev.jsonl" \
            --output_dir "$student_exp_dir"
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Stage 4: Train student with distillation and K-means quantization"
    resume_options=()
    if [ -n "$resume_checkpoint" ]; then
        resume_options=(--resume_from "$resume_checkpoint")
    fi

    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_distillation_kmeans.py \
            --seed "$seed" \
            --train_conf "$student_train_conf" \
            --train_file "$json_root/train.jsonl" \
            --eval_file "$json_root/dev.jsonl" \
            --output_dir "$student_exp_dir" \
            "${resume_options[@]}"
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Stage 5: Final K-means quantization and export"
    CUDA_VISIBLE_DEVICES="" \
        python local/export_kmeans_quantized_model.py \
            --exp_dir "$student_exp_dir"
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
    echo "Stage 6: Quantized student inference"
    mkdir -p "$student_exp_dir/test"
    CUDA_VISIBLE_DEVICES="$gpuid" \
        python finetuning/qwen3_asr_test.py \
            --exp_dir "$student_exp_dir/quantized-dense" \
            --input_jsonl "$json_root/test.jsonl" \
            --output_root "$student_exp_dir" \
            --device cuda:0 \
            --decoding_conf "$decoding_conf"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
    echo "Stage 7: MAC-SLU evaluation"
    python local/metrics.py \
        --output_dir "$student_exp_dir/test" \
        "$student_exp_dir/test/predictions.jsonl" \
        "$json_root/test.jsonl" \
        | tee "$student_exp_dir/test/metrics.txt"

    python local/plot_macslu_confusion.py \
        --pred_file "$student_exp_dir/test/predictions.jsonl" \
        --gt_file "$json_root/test.jsonl" \
        --labels_file "$labels_file" \
        --label_mapping_file "$label_mapping_file" \
        --output_dir "$student_exp_dir/test"
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
    echo "Stage 8: Compression and runtime report"
    benchmark_options=(--quantized_exp_dir "$student_exp_dir")
    if [ -n "$distillation_only_exp_dir" ]; then
        benchmark_options+=(--distillation_only_exp_dir "$distillation_only_exp_dir")
    fi
    python local/benchmark_macslu_quantized.py "${benchmark_options[@]}"
fi

if [ "$stage" -le 9 ] && [ "$stop_stage" -ge 9 ]; then
    echo "Stage 9: Summary and comparison"
    printf 'Vocabulary pruning enabled: %s\n' "$use_vocabulary_pruning"
    printf 'Vocabulary variant: %s\n' "$vocabulary_variant"
    printf 'Vocabulary manifest: %s\n' "$vocabulary_manifest"
    printf 'Teacher source checkpoint: %s\n' "$teacher_source_checkpoint"
    printf 'Teacher used for student: %s\n' "$teacher_for_student"
    printf 'Quantized student: %s\n' "$student_exp_dir"

    if [ -f "$student_exp_dir/compression_report.txt" ]; then
        cat "$student_exp_dir/compression_report.txt"
    fi
    if [ -n "$distillation_only_exp_dir" ]; then
        cat "$student_exp_dir/comparison_report.json"
    else
        echo "Cross-version comparison not requested."
    fi
fi
