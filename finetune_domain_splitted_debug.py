import copy
import os
import time
import torch
import wandb
from torchinfo import summary
import yaml
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset, registry
from src.eval import eval_single_dataset, eval_given_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder

PRINT_EVERY = 100

def interpolate_weights(theta_0, theta_1, alpha, fisher_mat=None):
    # interpolate between checkpoints with mixing coefficient alpha
    
    # weights of current task model, index: 1
    theta_0 = {k:v.cpu() for k,v in theta_0.items() if isinstance(v, torch.Tensor)}
    theta_1 = {k:v.cpu() for k,v in theta_1.items() if isinstance(v, torch.Tensor)}
    if fisher_mat is None:
        theta = {
            key: ((1 - alpha) * theta_0[key] + alpha * theta_1[key]) 
            for key in theta_0.keys() 
        }        
        return theta, None
    assert len(fisher_mat) == 2
    fisher_mat = [{k:v.cpu() for k,v in f.items() if isinstance(v, torch.Tensor) } for f in fisher_mat]
    F_theta1 = {
        key: fisher_mat[1].get(key, 1.) * theta_1[key]
        for key in theta_1.keys()
    }

    # weights of previous task model, index: 1
    F_theta0 = {
        key: fisher_mat[0].get(key, 1.)* theta_0[key]
        for key in theta_0.keys()
    }
    # inspect which keys are missing:
    # Weighted average of the weights using Fisher coeff, and normalize
    # new_theta = ((1 - alpha) * F0 *theta0 + alpha * F1 *theta1) / ((1 - alpha) * F0 + alpha * F1)
    new_F = {
        key: ((1 - alpha) * fisher_mat[0][key] + alpha * fisher_mat[1][key])
        for key in fisher_mat[0].keys()
    }

    theta = {
        key: ((1 - alpha) * F_theta0[key] + alpha * F_theta1[key]) / new_F.get(key, 1.)
        for key in theta_0.keys() 
    }
    # print("new_F", new_F.keys())
    # print("*"*100)
    
    # print("theta", theta.keys())
    # print("*"*100)

    # print("*"*100)
    # print("Merged", [key for key in theta_0.keys() if key in new_F.keys()])
    
    # print("*"*100)
    # print([key for key in theta_0.keys() if key not in new_F.keys()], "are missing")

    # Find the additional head weights
    # unique_keys = set(theta_1.keys()) - set(theta_0.keys())
    # for item in unique_keys:
    #     theta[item] = F_theta1[item]
    return theta, new_F



def finetune(args, eval_0shot=False, only_eval_0shot=False):
    if args.task_idx>1:
        exit()
    train_dataset_name = args.dataset
    dataset_class = registry[train_dataset_name].BASE_CLASS
    ckpdir = args.save
    subset_config_id = dataset_class.get_md5(args.subset_config)

    # Check if checkpoints already exist
    if args.sequential_finetuning:
        ft_path = os.path.join(ckpdir, f'checkpoint_{args.task_idx}.pt')
    else:
        ft_path = os.path.join(ckpdir, f'checkpoint_{subset_config_id}.pt')
    if os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return

    assert train_dataset_name is not None, "Please provide a training dataset."
    # if args.load is not None and args.load.endswith('pt'):
    #     image_encoder = ImageEncoder.load(args.load)
    # elif args.sequential_finetuning and args.task_idx:
    #     prev_ckpt = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{args.task_idx-1}.pt')
    #     print(f'Loading image encoder from prev task {prev_ckpt}')
    #     image_encoder = torch.load(prev_ckpt)
    # else:
    #     print('Building image encoder.')
    #     image_encoder = ImageEncoder(args, keep_lang=True)
        
    if args.load is not None and args.load.endswith('pt'):
        print(f'Loading previous  image encoder {args.load}.')
        image_encoder = ImageEncoder.load(args.load)
    elif args.sequential_finetuning and args.task_idx:
        prev_ckpt = os.path.join(ckpdir, f'checkpoint_{args.task_idx-1}.pt')
        print(f'Loading  linear image encoder from prev task {prev_ckpt}')
        image_encoder = ImageEncoder.load(prev_ckpt)
        # image_encoder.replace_params()
        prev_fisher = torch.load(os.path.join(ckpdir, f'fisher_{args.task_idx-1}.pt'))
    else:
        print('Building linear image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=True)        

    preprocess_fn = image_encoder.train_preprocess
    
    # # ZERO-SHOT EVAL ON EACH DOMAIN #
    # if not args.skip_eval:
    #     wandb.log({'subset_config_ID': subset_config_id})
        
    #     if eval_0shot or only_eval_0shot:
    #         _full_r = eval_single_dataset(image_encoder, train_dataset_name, args)['top1']
    #         wandb.log({f'full_acc': _full_r * 100.0})

    #         for domain in dataset_class.DOMAINS:
    #             _subset_config = {
    #                 'domains': [domain],
    #                 'classes': dataset_class.CLASSES
    #             }
    #             _dataset = get_dataset(
    #                 train_dataset_name,
    #                 preprocess_fn,
    #                 location=args.data_location,
    #                 batch_size=args.batch_size,
    #                 subset_config=_subset_config,
    #             )
    #             _r = eval_given_dataset(image_encoder, _dataset, train_dataset_name, args)['top1']

    #             wandb.log({f'{domain}_acc': _r * 100.0})
        
    if only_eval_0shot:
        return
    ##################
    
    dataset = get_dataset(
        train_dataset_name,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        subset_config=args.subset_config,
    )
    subset_config_0 = {
        'domains': [dataset_class.DOMAINS[dataset_class.default_domain_order[0]]],
        'classes': dataset_class.CLASSES,
        'domain_idx': domain_idx,
    }
    dataset_0 = get_dataset(
        train_dataset_name,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        subset_config=subset_config_0,
    )

    subset_config_1 = {
        'domains': [dataset_class.DOMAINS[dataset_class.default_domain_order[1]]],
        'classes': dataset_class.CLASSES,
        'domain_idx': domain_idx,
    }
    dataset_1 = get_dataset(
        train_dataset_name,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        subset_config=subset_config_1,
    )
    
    if not args.skip_eval:
        wandb.log({
            'train_subset_samples': len(dataset.train_dataset),
            'test_subset_samples': len(dataset.test_dataset),
        })

    classification_head = get_classification_head(args, train_dataset_name, classnames=dataset.classnames)
    # model = ImageClassifier(image_encoder, classification_head)
    # model.freeze_head()

    prev_image_encoder = copy.deepcopy(image_encoder)
    prev_model = ImageClassifier(prev_image_encoder, classification_head)
    model = ImageClassifier(image_encoder, classification_head)
    for n, p in classification_head.named_parameters():
        p.requires_grad  = False
    model.freeze_head()
    # model.freeze_lang()
    print(summary(model))
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = model.cuda()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    num_batches = len(dataset.train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
    data_loader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=None)
    for epoch in range(args.epochs):
        
        model = model.cuda()
        model.train()
        
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)
        n_batches = len(data_loader)
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time
            wandb.log({
                f"task_idx_{args.task_idx}_train_loss": loss.item(),
                "epoch": epoch,
                "step": step
            })
            if step % PRINT_EVERY == 0 or i + 1 == n_batches:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        # break        

    print("Collecting FIM")
    fisher = {}
    # Iterate over the model's named parameters
    for name, param in model.image_encoder.named_parameters():
        # Only consider parameters that require gradients
        if param.requires_grad:
        # The optimizer state dict uses the parameter tensor itself as a key.
            state = optimizer.state.get(param, {})
            # Check if the state contains the "exp_avg_sq" value
            if 'exp_avg_sq' in state:
                # Clone and detach so that the stored tensor is not connected to the graph
                fisher[name] = state['exp_avg_sq'].clone().detach()
            else:
                print(f"No exp_avg_sq for parameter: {name}")
    state_dict = copy.deepcopy(model.image_encoder.state_dict())
    if args.task_idx>0:
        print("Start merging with Fisher")
        alpha_merge_coefs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        
        for alpha_merge in alpha_merge_coefs:
            new_state_dict, _ = interpolate_weights(prev_image_encoder.state_dict(), state_dict,
                                                alpha=alpha_merge,
                                                fisher_mat=None)
            model.image_encoder.load_state_dict(new_state_dict)
            # _full_r = eval_single_dataset(image_encoder, train_dataset_name, args)['top1']
            # wandb.log({f'full_acc': _full_r * 100.0, "linear":alpha_merge})
            # print("Coef", alpha_merge, "Linear Acc", _full_r)     

            acc_0 = eval_given_dataset(image_encoder, dataset_0, train_dataset_name, args)['top1']   
            wandb.log({f'acc_0': acc_0 * 100.0, "linear":alpha_merge})
            acc_1 = eval_given_dataset(image_encoder, dataset_1, train_dataset_name, args)['top1']   
            wandb.log({f'acc_1': acc_1 * 100.0, "linear":alpha_merge})           
            wandb.log({f'acc_01': (acc_1+acc_0) * 50.0, "linear":alpha_merge})                                     
                                      
        for alpha_merge in alpha_merge_coefs:
            new_state_dict, _ = interpolate_weights(prev_image_encoder.state_dict(), state_dict,
                                                alpha=alpha_merge,
                                                fisher_mat=[prev_fisher,fisher])
            model.image_encoder.load_state_dict(new_state_dict)
            # _full_r = eval_single_dataset(image_encoder, train_dataset_name, args)['top1']
            # wandb.log({f'full_acc': _full_r * 100.0, "fisher":alpha_merge})
            # print("Coef", alpha_merge, "Fisher Acc", _full_r)
            acc_0 = eval_given_dataset(image_encoder, dataset_0, train_dataset_name, args)['top1']   
            wandb.log({f'acc_0': acc_0 * 100.0, "fisher":alpha_merge})
            acc_1 = eval_given_dataset(image_encoder, dataset_1, train_dataset_name, args)['top1']   
            wandb.log({f'acc_1': acc_1 * 100.0, "fisher":alpha_merge})           
            wandb.log({f'acc_01': (acc_1+acc_0) * 50.0, "fisher":alpha_merge})         
        
    if args.save is not None:
        image_encoder.save(ft_path)
        torch.save(fisher, os.path.join(ckpdir, f'fisher_{args.task_idx}.pt'))



if __name__ == '__main__':

    args = parse_arguments()

    args.model = 'ViT-B-16'
    args.batch_size = 128
    dataset_class = registry[args.dataset]
    args.save = "outputs/dil/debug-2/" + str(args).replace(", ", "/").replace(
        "'", ""
    ).replace("(", "").replace(")", "").replace("Namespace", "")

    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save , "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    wandb.init(project="DIL-debug-2", config=vars(args))

    for task_idx, domain_idx in enumerate(dataset_class.default_domain_order):
        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
            'classes': dataset_class.BASE_CLASS.CLASSES,
            'domain_idx': domain_idx,
        }
        
        args.task_idx = task_idx if args.sequential_finetuning else None


        print('='*100)
        print(f'Finetuning {args.model} on {args.dataset}')
        print('='*100)

        finetune(args)
