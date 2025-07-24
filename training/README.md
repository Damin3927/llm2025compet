# 学習　Training

## 環境構築
HOMEで以下を実行する。venvが作成されライブラリがインストールされる。
```bash
$ . ./llm2025compet/training/create_open-r1_env.sh
```

## 学習の実行

### bashで実行
HOMEで以下を実行する。
```bash
$ cd ~
$ srun --partition P02 --nodes=1 --nodelist osk-gpu[56] --gpus-per-node=1 --time 1:00:00 --pty bash -i
$ . ./llm2025compet/training/commands/sft-qwen-0.5b.sh
```

### sbatchで実行
HOMEで以下を実行する。
```bash
$ cd ~
$ sbatch ./llm2025compet/training/commands/sft-qwen-0.5b.sh
```
