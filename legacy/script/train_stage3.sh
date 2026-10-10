output_dir=out/stage3_finetuning

# RGB-pose setting
ckpt_path=out/stage2_pretraining/best_checkpoint.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --rgb_support # enable RGB-pose setting

# example of ISLR
# deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#    --batch-size 8 \
#    --gradient-accumulation-steps 1 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir $output_dir \
#    --finetune $ckpt_path \
#    --dataset WLASL \
#    --task ISLR \
#    --max_length 64 \
#    --rgb_support # enable RGB-pose setting

## pose only setting
#ckpt_path=out/stage1_pretraining/best_checkpoint.pth
#
#deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#  --batch-size 8 \
#  --gradient-accumulation-steps 1 \
#  --epochs 20 \
#  --opt AdamW \
#  --lr 3e-4 \
#  --output_dir $output_dir \
#  --finetune $ckpt_path \
#  --dataset CSL_Daily \
#  --task SLT \
##   --rgb_support # enable RGB-pose setting