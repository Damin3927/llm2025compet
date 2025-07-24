module load cuda/12.8
source openr1/bin/activate
cd llm2025compet/training/open-r1/src

srun --partition P02 --nodes=1 --nodelist osk-gpu[56] --gpus-per-node=1 --pty bash -i
. ../../tutorials/sft-0.5b.sh

# sbatch --partition P02 --nodes=1 --nodelist osk-gpu[56] --gpus-per-node=1 --job-name sft-0.5b --output sft-0.5b.out --error sft-0.5b.err --wrap "bash -i -c 'source ../../tutorials/sft-0.5b.sh'"

# . ./llm2025compet/training/start_train.sh

# ターミナルの初期状態からsftを実行するまでの一連のコマンド
# ターミナルを新規作成した後、これを実行すると、環境を整えてからsftのサンプルコードを実行できます。
# GPU設定やSFT実行コマンドは自由に変更してください。

# おそらく、srunで止まるので、別ターミナルを用いて以下のコマンドでダミージョブをキャンセルしてください。
# /home/Competition2025/P02/shareP02/scripts/scancel.sh 287614