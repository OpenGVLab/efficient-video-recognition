#!/usr/bin/env sh

exp_dir=runs/k400_vitb16_32f_dec4x768

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 8 \
  main.py \
    --num_steps 50000 \
    --backbone "ViT-B/16-lnpre" \
    --backbone_type clip \
    --backbone_path /path/to/clip_models/ViT-B-16.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 768 \
    --decoder_num_heads 12 \
    --num_classes 400 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --train_list_path /path/to/k400/train.txt \
    --val_list_path /path/to/k400/val.txt \
    --batch_size 256 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 12 \
    --num_frames 32 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
