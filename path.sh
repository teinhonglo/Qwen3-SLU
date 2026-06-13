export PYTHONNOUSERSITE=1
#export WANDB_DISABLED=true
#export WANDB_MODE=offline
#export PYTHONPATH="."

KALDI_ROOT=/share/nas167/teinhonglo/espnets/espnet-2025/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
#eval "$(conda shell.bash hook)"
CUDA_DIR=/usr/local/cuda

if [ -d $CUDA_DIR ]; then
    export PATH=$CUDA_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
fi
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate qwen3-slu
