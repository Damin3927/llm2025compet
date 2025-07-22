# open-r1の学習コードを実行するための環境を構築する。
# 基本的にリポジトリのREADME.mdに従う。
# venvを使う

uv venv openr1 # venvの作成
source openr1/bin/activate # venvの有効化
uv pip install --upgrade pip # pipのアップグレード

# vllmのインストール
uv pip install vllm==0.8.5.post1 
# flash-attnのインストール
uv pip install setuptools && uv pip install flash-attn --no-build-isolation 

# open-r1及び、他のライブラリのインストール
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "llm2025compet/training/open-r1[dev]" 