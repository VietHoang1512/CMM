#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 -C 'a100|rtx8000'
#SBATCH --time=20:00:00
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
lwf_lambda=0.3
ewc_lambda=1e6
epochs=10
# ImageNetR
representation_lr=0.

for dataset in  Office31 ; do
    # out_dir=outs/${model}/sequential_finetuning/domain_incremental/ft/${dataset}
    # mkdir -p ${out_dir}

    # python finetune_domain_splitted.py \
    # --model ${model} \
    # --dataset ${dataset} \
    # --epochs ${epochs} \
    # --sequential-finetuning \
    #     |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out


    # out_dir=outs/${model}/sequential_finetuning/domain_incremental/merging/${dataset}
    # mkdir -p ${out_dir}

    # python merge_domain_splitted.py \
    # --model ${model} \
    # --dataset ${dataset} \
    # --epochs ${epochs} \
    # --sequential-finetuning \
    #     |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out
    # bash scripts/DIL/ewc.sh ${model} ${dataset} ${epochs} ${ewc_lambda} ${seed}
    # bash scripts/DIL/lwf.sh ${model} ${dataset} ${epochs} ${lwf_lambda} ${seed}
    for alpha_merge in .6 ; do
        python finetune_domain_splitted_linear.py \
        --model ${model} \
        --dataset ${dataset} \
        --epochs ${epochs} \
        --sequential-finetuning \
        --representation_loss l2 \
        --representation_lr $representation_lr \
        --sequential-finetuning \
        --seed ${seed} \
        --alpha_merge $alpha_merge 
    done



    # alpha_merge=0.
    # python finetune_domain_splitted_linear_debug.py \
    # --lr 1e-3 \
    # --model ${model} \
    # --dataset ${dataset} \
    # --epochs ${epochs} \
    # --sequential-finetuning \
    # --representation_loss l1 \
    # --representation_lr $representation_lr \
    # --sequential-finetuning \
    # --seed ${seed} \
    # --alpha_merge $alpha_merge 

done
