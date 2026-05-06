#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config
repo_id="Gatsby1984/MAC_SLU"
data_root="data/macslu"
download_dir=${data_root}/raw
extract_root=${data_root}/audio
json_root=data-json/macslu
fixed_json_root=data-json/macslu_fixed
prompt_file=""   # 可指定外部 prompt 檔案，空字串則使用 prepare_macslu_jsonl.py 內建 prompt

# stage
stage=0
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download MAC-SLU and prepare jsonl"

    if [ -d "$json_root" ] && [ -f "$json_root/train.jsonl" ] && [ -f "$json_root/dev.jsonl" ] && [ -f "$json_root/test.jsonl" ]; then
        echo "[INFO] Skip Stage 0 because $json_root already exists with train/dev/test JSONL"
    else
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
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Audit/fix MAC-SLU JSONL"

    python local/audit_and_fix_macslu_jsonl.py \
        --jsonl-root "$json_root" \
        --splits train dev test \
        --output-dir "$fixed_json_root"
fi
