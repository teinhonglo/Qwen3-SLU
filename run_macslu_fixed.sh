#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa, huggingface_hub

set -euo pipefail

stage=1
stop_stage=2

. ./local/parse_options.sh
. ./path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -f data-json/macslu/train.jsonl ] && \
       [ -f data-json/macslu/dev.jsonl ] && \
       [ -f data-json/macslu/test.jsonl ]; then
        echo "[Stage 1] Existing jsonl files found. Skip prepare_macslu_jsonl.py."
    else
        echo "[Stage 1] Preparing MacSLU jsonl..."
        python local/prepare_macslu_jsonl.py \
            --repo-id "Gatsby1984/MAC_SLU" \
            --download-dir "data/macslu/raw" \
            --extract-root "data/macslu/audio" \
            --jsonl-root "data-json/macslu" \
            --splits train dev test
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "[Stage 2] Fixing MacSLU jsonl annotations..."
    python local/fix_macslu_jsonl.py \
        --input_dir data-json/macslu \
        --output_dir data-json/macslu_fixed \
        --labels_path data-json/macslu/labels.txt \
        --splits train dev test \
        --write_reports
fi
