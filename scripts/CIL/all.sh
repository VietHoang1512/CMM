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
set -e
model=ViT-B-16
seed=5
lwf_lambda=0.3
ewc_lambda=1e6
representation_lr=1e-7


args=("$@")
echo $# arguments passed
echo "alpha_merge" ${args[0]} "n_splits" ${args[1]} 
alpha_merge=${args[0]}
n_splits=${args[1]}

alpha_merge=.4
epochs=10
# dataset=CIFAR100
# for alpha_merge in  .6  .5 .4   .3  ; do
#     for n_splits in  10  20 50 ; do

#         python finetune_splitted_linear.py \
#             --model ${model} \
#             --dataset ${dataset} \
#             --epochs ${epochs} \
#             --n_splits ${n_splits} \
#             --split_strategy class \
#             --representation_loss l2 \
#             --representation_lr $representation_lr \
#             --sequential-finetuning \
#             --seed ${seed} \
#             --alpha_merge $alpha_merge 
#     done
# done

epochs=10
dataset=ImageNetR
for alpha_merge in .4 .5 .3 ; do
    for n_splits in 10    ; do

        python finetune_splitted_linear.py \
            --model ${model} \
            --dataset ${dataset} \
            --lr 1e-5 \
            --model 'ViT-B-16' \
            --epochs ${epochs} \
            --n_splits ${n_splits} \
            --representation_loss l2 \
            --representation_lr $representation_lr \
            --split_strategy class \
            --sequential-finetuning \
            --seed ${seed} \
            --alpha_merge $alpha_merge 
    done
done



# epochs=30
# dataset=CUB200
# alpha_merge=.4
# for alpha_merge in .4 .5 .3 ; do
#     for n_splits in  5 20; do
#         # TRAIN


#         python finetune_splitted_linear.py \
#             --model ${model} \
#             --dataset ${dataset} \
#             --epochs ${epochs} \
#             --n_splits ${n_splits} \
#             --representation_lr $representation_lr \
#             --representation_loss l2 \
#             --split_strategy class \
#             --sequential-finetuning \
#             --seed ${seed} \
#             --alpha_merge $alpha_merge 

#     done
# done


# epochs=30
# dataset=Cars
# for alpha_merge in .4 .5 .3 ; do
#     for n_splits in 5 10 20 ; do


#             python finetune_splitted_linear.py \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --epochs ${epochs} \
#                 --n_splits ${n_splits} \
#                 --split_strategy class \
#                 --representation_lr $representation_lr \
#                 --sequential-finetuning \
#                 --seed ${seed} \
#                 --alpha_merge $alpha_merge 
#     done
# done
pip install matplotlib scikit-learn
representation_lr=1e-5
epochs=30
dataset=Cars
for alpha_merge in .3 ; do
    for n_splits in 5  ; do
            python finetune_splitted_linear_debug.py \
                --model ${model} \
                --dataset ${dataset} \
                --epochs ${epochs} \
                --lr 1e-4 \
                --n_splits ${n_splits} \
                --split_strategy class \
                --representation_loss l1 \
                --representation_lr $representation_lr \
                --sequential-finetuning \
                --seed ${seed} \
                --alpha_merge $alpha_merge 
    done
done