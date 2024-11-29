torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --data_root_dir /raid/datasets/openx/ \
  --run_root_dir /raid/users/homer/vla-distill-output/ \
  --wandb_entity homer