#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  -C 'a100|rtx8000'
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=cl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
export PYTHONPATH=$PWD

model=ViT-B-16
seed=5

# bash scripts/8ds/finetune_8ds.sh ${model} ${seed}
# bash scripts/8ds/finetune_8ds_seq.sh ${model} ${seed}

# TRAIN
out_dir=outs/${model}/8datasets/ft/seq
mkdir -p ${out_dir}

python finetune_8datasets.py \
    --model ${model} \
    --sequential-finetuning \
    --seed ${seed} \
        |& tee ${out_dir}/seed:${seed}.out


# MERGE
out_dir=outs/${model}/8datasets/merging/seq
mkdir -p ${out_dir}

python merge_8datasets.py \
    --model ${model} \
    --sequential-finetuning \
    --seed ${seed} \
        |& tee ${out_dir}/seed:${seed}.out
