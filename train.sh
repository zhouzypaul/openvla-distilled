# bash script
export PRISMATIC_DATA_ROOT="/scratch/partial_datasets/openx/bridge_orig"
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-bridge \
  --data_root_dir /scratch/partial_datasets/openx \
  --run_root_dir /shared/projects/icrl/openvla_out \
  --run_id scratch_prism-qwen25-dinosiglip-224px+0_5b+mx-bridge \
  --image_aug True \
  --wandb_project openvla_distill \
  --wandb_entity project_vit \
  --is_resume False
