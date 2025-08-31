#!/bin/bash

deepspeed --master_port 29650 --include localhost:0,2,3,4,5,6,7,8 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version qwen_2_5 \
    --freeze_backbone False \
    --data_path playground/data/InternVL-Chat-V1-2-SFT-Data/merged_sft_data.json \
    --image_folder playground/data/InternVL-Chat-V1-2-SFT-Data/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --llm_init_from_path /home/jusheng/yijia/LLaVA_HCP/checkpoints/llava-v1.5-7b \
    --save_full_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-sft-stage2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain-0814/checkpoint-48000/mm_projector.bin \
    --resume_from_checkpoint ./checkpoints/llava-v1.5-7b-sft-stage2/checkpoint-45000


