#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

# data config (overridable via parse_options)
repo_id="Gatsby1984/MAC_SLU"
download_dir="data/macslu/raw"
extract_root="data/macslu/audio"
jsonl_root="data-json/macslu"
fixed_output_dir="data-json/macslu_fixed"
labels_path="data-json/macslu/labels.txt"
splits="train dev test"

stage=1
stop_stage=2

. ./local/parse_options.sh
. ./path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -f "${jsonl_root}/train.jsonl" ] && \
       [ -f "${jsonl_root}/dev.jsonl" ] && \
       [ -f "${jsonl_root}/test.jsonl" ]; then
        echo "[Stage 1] Existing jsonl files found. Skip prepare_macslu_jsonl.py."
    else
        echo "[Stage 1] Preparing MacSLU jsonl..."
        python local/prepare_macslu_jsonl.py \
            --repo-id "${repo_id}" \
            --download-dir "${download_dir}" \
            --extract-root "${extract_root}" \
            --jsonl-root "${jsonl_root}" \
            --splits ${splits}
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "[Stage 2] Fixing MacSLU jsonl annotations..."
    python local/fix_macslu_jsonl.py \
        --input_dir "${jsonl_root}" \
        --output_dir "${fixed_output_dir}" \
        --labels_path "${labels_path}" \
        --splits ${splits} \
        --write_reports
fi
