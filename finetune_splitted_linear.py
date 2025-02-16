import os
import time
import torch
import copy
import wandb
import random
import numpy as np
from torchinfo import summary

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.merging.task_vectors import TaskVector
from src.linearize import LinearizedImageEncoder
from src.eval import evaluate,  eval_single_dataset

def interpolate_weights(theta_0, theta_1, alpha, fisher_mat):
    # interpolate between checkpoints with mixing coefficient alpha
    assert len(fisher_mat) == 2
    # weights of current task model, index: 1
    theta_0 = {k:v.cpu() for k,v in theta_0.items() if isinstance(v, torch.Tensor)}
    theta_1 = {k:v.cpu() for k,v in theta_1.items() if isinstance(v, torch.Tensor)}
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


def finetune(args):
    train_dataset = args.dataset
    ckpdir = args.save

    # finetune for each split separately
    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f'finetuned_{split_idx}.pt')
        if os.path.exists(os.path.join(ckpdir, f'finetuned_{split_idx}.pt')):
            print(f"Skipping finetuning on split {split_idx}, "
                  f"ckpt already exists under {os.path.join(ckpdir, f'finetuned_{split_idx}.pt')}")
            continue
            
            # if split_idx==0:
            #     continue

        assert train_dataset is not None, "Please provide a training dataset."
        if args.load is not None and args.load.endswith('pt'):
            print(f'Loading previous linear image encoder {args.load}.')
            image_encoder = LinearizedImageEncoder.load(args.load)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f'finetuned_{split_idx-1}.pt')
            print(f'Loading  linear image encoder from prev task {prev_ckpt}')
            image_encoder = LinearizedImageEncoder.load(prev_ckpt)
            # image_encoder.replace_params()
            prev_fisher = torch.load(os.path.join(ckpdir, f'fisher_{split_idx-1}.pt'))
        else:
            print('Building linear image encoder.')
            image_encoder = LinearizedImageEncoder(args)

        if split_idx==0 and not os.path.exists(f'checkpoints/{args.model}/zeroshot.pt'):
            image_encoder.save(f'checkpoints/{args.model}/zeroshot.pt')
        preprocess_fn = image_encoder.train_preprocess
        print_every = 100

        dataset = get_dataset(
            train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, image_encoder, args
        )
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
        # model = torch.nn.DataParallel(model, device_ids=devices)
        model = model.to("cuda:0")
        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        num_batches = len(dataset.train_loader)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        n_batches = len(data_loader)

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)

        for epoch in range(args.epochs):
            model = model.cuda()
            model.train()

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
                # for p in params:
                #     print(p.grad)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                # print(model.image_encoder.get_trainable_params().grad)
                optimizer.step()
                batch_time = time.time() - start_time
                
                # Log the training loss at every step for the current split
                wandb.log({
                    f"split_{split_idx}_train_loss": loss.item(),
                    "epoch": epoch,
                    "step": step
                })
                
                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )
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
        state_dict = model.image_encoder.state_dict()
        if split_idx>0:
            print("Start merging")
            state_dict, fisher = interpolate_weights(prev_image_encoder.state_dict(), state_dict,
                                                alpha=args.alpha_merge,
                                                fisher_mat=[prev_fisher,fisher])
        # Evaluate
        model.image_encoder.load_state_dict(state_dict)
        if split_idx>0:
            print("Start representation finetuning")
            model = model.cuda()
            # model.image_encoder.freeze_backbone()
            model.train()
            # print(summary(model))
            
            prev_model = prev_model.to("cuda:0")
            prev_model.eval()
            params = [p for p in model.parameters() if p.requires_grad]
            params = [model.image_encoder.get_trainable_params()]
            print("Optimized params", params)
            optimizer = torch.optim.AdamW(params, lr=args.representation_lr)
            if args.representation_loss == "l1":
                loss_func = torch.nn.L1Loss()
            elif args.representation_loss == "l2":
                loss_func = torch.nn.MSELoss()
            else:
                raise NotImplementedError
            for param in prev_model.parameters():
                param.requires_grad = False
            for i, batch in enumerate(data_loader):

                model.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time

                _, cur_features = model(inputs, return_features=True)
                _, prev_features = prev_model(inputs, return_features=True)

                loss = loss_func(cur_features, prev_features)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                print(loss.item())
                # for n, p in model.named_parameters():
                #     if p.requires_grad and p.grad is not None:
                #         try:
                #             print(n, max(p.grad))  
                #         except:
                #             pass              
                
                # for p in params:
                #     print((p.grad))
                optimizer.step()                
                wandb.log({
                    f"split_{split_idx}_representation_loss": loss.item(),
                    "representation_step": i,
                    })

            # model.image_encoder.freeze_backbone()
            print("Optimized params", params)
            
        image_encoder = model.image_encoder
        result = eval_single_dataset(image_encoder, args.dataset, args) 
        print("Results: ", result)
        wandb.log({"split": split_idx, "test_accuracy": result['top1']})



        if args.save is not None:
            # save optimizer state
            torch.save(fisher, os.path.join(ckpdir, f'fisher_{split_idx}.pt'))
            image_encoder.save(ft_path)
        

if __name__ == '__main__':
    args = parse_arguments()

# -----------------------------
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    args.lr = 1e-5
    args.batch_size = 128
    # sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    # args.save = f'checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental'
    args.save = "outputs/cil/linear/" + str(args).replace(", ", "/").replace(
        "'", ""
    ).replace("(", "").replace(")", "").replace("Namespace", "")

    os.makedirs(args.save, exist_ok=True)
    wandb.init(project="CIL-linear", config=vars(args))
    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits)')
    print('='*100)

    finetune(args)
