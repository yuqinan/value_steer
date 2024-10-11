#!/bin/bash
#SBATCH --partition=tibet --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4
#SBATCH --nodelist=sphinx6
#SBATCH --open-mode=append 
#SBATCH --partition=sphinx
#SBATCH --job-name=test_training
#SBATCH --output=sample_story.out
#SBATCH --account=nlp

# only use the following if you want email notification
####SBATCH --mail-user=qinanyu@stanford.edu
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source activate alpaca

torchrun --nproc_per_node=4 train.py \
    --model_name_or_path  meta-llama/Llama-3.1-8B \
    --data_path ./abortion.json \
    --bf16 True \
    --output_dir /nlp/scr/qinanyu/model_cache/result_alpaca_abortion \
    --num_train_epochs 3 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done