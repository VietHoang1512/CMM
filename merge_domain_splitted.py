import numpy as np
import torch
import wandb
import os

from src.merging.task_vectors import TaskVector, merge_max_abs, last, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_given_dataset

from src.args import parse_arguments
from src.datasets.registry import get_dataset, registry
from src.modeling import ImageEncoder, ImageClassifier


# Config
args = parse_arguments()

args.model = 'ViT-B-16'
pretrained_checkpoint = f'checkpoints/{args.model}/dil_zeroshot.pt'

image_encoder = ImageEncoder(args, keep_lang=True)
image_encoder.save(pretrained_checkpoint)

def search_evaluate_merging(dataset, task_vectors, args, n_coeffs=20):
    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/n_splits, num=n_coeffs+1)[1:]),
        (last, [.5, 1.0]),
        (sum, [1.0/args.n_splits]),
        (merge_rnd_mix, [1.0]),
        (merge_max_abs, [0.5]),
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        merged_tv = f(task_vectors)
        
        # Apply the resulting task vector
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            
            dataset_class = (registry[args.dataset]).BASE_CLASS
            print("Merging with", f, scaling_coef)
            for domain in dataset_class.DOMAINS:
                _subset_config = {
                    'domains': [domain],
                    'classes': dataset_class.CLASSES
                }
                _dataset = get_dataset(
                    args.dataset,
                    image_encoder.train_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size,
                    subset_config=_subset_config,
                )
                _r = eval_given_dataset(image_encoder, _dataset, args.dataset, args)['top1']
                print("domain",domain, "acc", _r)
            # Evaluate
            _r = eval_single_dataset(image_encoder, dataset, args)['top1']
            wandb.log({
                f"merging/{f.__name__}": _r * 100.0,
                "helpers/merging/alpha": scaling_coef,
            })
            results[scaling_coef] = _r

        print(f"Results with function {f.__name__}:\n{results}")
        
        
    # TIES merging
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    
    print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
    
    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )
    merged_tv = TaskVector(vector=merged_tv)

    # Apply the resulting task vector
    results = {}
    # for scaling_coef in np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]:
    for scaling_coef in [0.55]:
        print(f"Scaling coeff: {scaling_coef}")
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        # Evaluate
        _r = eval_single_dataset(image_encoder, dataset, args)['top1']
        wandb.log({
            f"merging/TIES": _r * 100.0,
            "helpers/merging/alpha": scaling_coef,
        })
        results[scaling_coef] = _r
            
    print(f"Results with function TIES:\n{results}")



if __name__ == '__main__':    

    task_vectors = []
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    name=f"merging-{args.dataset}-DIL-{method}"

    wandb.init(
        project="DIL",
        config=vars(args),
    )

    for task_idx, domain_idx in enumerate(dataset_class.default_domain_order):
        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }
        subset_config_id = dataset_class.BASE_CLASS.get_md5(args.subset_config)
        if args.sequential_finetuning:
            args.save = f'checkpoints/{args.model}/sequential_finetuning/domain_incremental'
            ckpdir = os.path.join(args.save, args.dataset)
            ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{task_idx}.pt')
            ft_path = f"outputs/dil/linear/data_location=/vast//data/dataset=Office31/eval_datasets=None/train_dataset=None/exp_name=None/model=ViT-B-16/batch_size=128/lr=1e-05/representation_lr=0.0/representation_loss=l1/wd=0.1/ls=0.0/warmup_length=500/epochs=10/skip_eval=False/load=None/save=None/results_db=None/cache_dir=None/openclip_cachedir=checkpoints/ViT-B-16/cachedir/open_clip/n_splits=2/split_strategy=None/sequential_finetuning=True/lwf_lamb=0.0/ewc_lamb=0.0/alpha_merge=0.6/seed=5/wandb_entity_name=YOUR-WANDB-ACCOUNT/device=cuda/checkpoint_{task_idx}.pt"
        else:
            args.save = f'checkpoints/{args.model}/partial_datasets'
            ckpdir = os.path.join(args.save, args.dataset)
            ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{subset_config_id}.pt')
        
        tv = TaskVector(pretrained_checkpoint, ft_path)
        task_vectors.append(tv)

    print(f"\nEVAL: {args.dataset} - domain incremental")    
    search_evaluate_merging(args.dataset, task_vectors, args)
