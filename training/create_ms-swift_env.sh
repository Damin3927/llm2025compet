# open-r1の学習コードを実行するための環境を構築する。
# 基本的にリポジトリのREADME.mdに従う。
# venvを使う

python -m venv ms-swift # venvの作成
source ms-swift/bin/activate # venvの有効化
pip install --upgrade pip # pipのアップグレード

module load cuda/12.8

cd llm2025compet/training/ms-swift || exit 1

pip install -e .

pip install liger-kernel transformers -U

pip install flash-attn==2.5.8 --no-build-isolation 

pip install pybind11

export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && echo $SITE_PACKAGES && \
export CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include && \
export LD_LIBRARY_PATH=/home/appli/cuda/12.8/lib64 && \
export CUDA_HOME=/home/appli/cuda/12.8 && \
export CUDNN_LIBRARY=/home/appli/cuda/12.8/lib64 && \
export CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include && \
export TORCH_CUDA_ARCH_LIST="9.0"

# nvidia-smi --query-gpu=compute_cap --format=csv,noheader # GPUのCompute Capabilityを調べる

if [ ":$PATH:" != *":/home/appli/cuda/12.8/bin:"* ]; then
    export PATH="/home/appli/cuda/12.8/bin:$PATH"
fi
echo "PATH ... $PATH"

pip install --no-build-isolation transformer_engine[pytorch]
#pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]

git clone https://github.com/NVIDIA/apex

cd apex

git checkout e13873debc4699d39c6861074b9a3b2a02327f92

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0

# If you are using multi-node training, please additionally set the `MODELSCOPE_CACHE` environment variable to a shared storage path.
# This will ensure that the dataset cache is shared, thereby speeding up preprocessing.
# export MODELSCOPE_CACHE='.cache/modelscope'

# Megatron-LM
# The training module in the dependent library Megatron-LM will be cloned and installed by swift via `git clone`. Alternatively, you can use the environment variable `MEGATRON_LM_PATH` to point to the path of an already downloaded repository (in offline environments, use the [core_r0.13.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.13.0)).
# export MEGATRON_LM_PATH='/xxx/Megatron-LM'

