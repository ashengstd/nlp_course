#!/bin/bash

mkdir -p './ckpt'
# 默认char
tokenizer_type=${tokenizer_type:-char}
# 根据 tokenizer_type 选择 vocab 参数
if [ "$tokenizer_type" = "char" ]; then
    vocab_path="./model/vocab.txt"
elif [ "$tokenizer_type" = "bpe" ]; then
    vocab_path="./model/m.model"
fi
CUDA_VISIBLE_DEVICES=0 \
python -u pretrain.py --embed_dim 768 \
                      --ff_embed_dim 3072 \
                      --num_heads 12 \
                      --layers 12 \
                      --dropout 0.2 \
                      --train_data ./data/sft/train_converted.txt \
                      --dev_data ./data/sft/val_converted.txt \
                      --vocab "$vocab_path" \
                      --min_occur_cnt 1 \
                      --batch_size 20 \
                      --warmup_steps 10000 \
                      --lr 1 \
                      --weight_decay 0 \
                      --smoothing 0.1 \
                      --max_len 256 \
                      --min_len 10 \
                      --world_size 1 \
                      --gpus 1 \
                      --start_rank 0 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 28888 \
                      --print_every 100 \
                      --save_every 5000 \
                      --epoch 100 \
                      --save_dir ckpt \
                      --backend nccl \
                      --tokenizer_type "$tokenizer_type" \
                      --start_from "./epoch0_batch_10000"