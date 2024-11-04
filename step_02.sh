mkdir -p './ckpt'

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u pretrain.py --embed_dim 768 \
                      --ff_embed_dim 3072 \
                      --num_heads 12 \
                      --layers 12 \
                      --dropout 0.2 \
                      --train_data ./data/train.txt \
                      --dev_data ./data/val_tiny.txt \
                      --vocab ./model/vocab.txt \
                      --min_occur_cnt 1 \
                      --batch_size 40 \
                      --warmup_steps 10000 \
                      --lr 1 \
                      --weight_decay 0 \
                      --smoothing 0.1 \
                      --max_len 256 \
                      --min_len 10 \
                      --world_size 4 \
                      --gpus 4 \
                      --start_rank 0 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 28512 \
                      --print_every 100 \
                      --save_every 10000 \
                      --epoch 100 \
                      --save_dir ckpt \
                      --backend nccl