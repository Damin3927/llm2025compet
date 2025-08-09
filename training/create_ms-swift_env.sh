# open-r1の学習コードを実行するための環境を構築する。
# 基本的にリポジトリのREADME.mdに従う。
# venvを使う

python -m venv ms-swift # venvの作成
source ms-swift/bin/activate # venvの有効化
pip install --upgrade pip # pipのアップグレード

module load cuda/12.8

#cd llm2025compet/training/ms-swift || exit 1

#pip install -e .

#pip install liger-kernel transformers -U