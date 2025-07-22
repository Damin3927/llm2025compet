accelerate launch \
    --config_file ../recipes/accelerate_configs/zero2.yaml --num_processes 1\
    open_r1/sft.py \
    --config ../../configs/Qwen2.5-Distill-0.5b-test/sft/config_distill.yaml
    #--config ../recipes/OpenR1-Distill-7B/sft/config_distill.yaml

# 1. cd llm2025compet/training/open-r1/src ここで実行すること
# 2. srun_bash.sh を実行、計算ノードに入る (sbatchでも可)
# 3. . ../../tutorials/sft-0.5b.sh

# 複数GPUならzero3.yamlを使う
# GPUが1つならzero2.yamlを使う

# 動作確認済み