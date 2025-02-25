import torch
import torch.nn as nn
from model2 import GPT
import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import set_seed
import wandb
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--folder', type=str, default="arithmetic")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--chain', action='store_true', default=False)
parser.add_argument('--rpe', action='store_true', default=False)
parser.add_argument('--maxlen', type=int, default=120)
parser.add_argument('--maxlen_ood', type=int, default=120)
parser.add_argument('--maxlen_IID1', type=int, default=120)
parser.add_argument('--maxlen_IID2', type=int, default=120)
parser.add_argument('--maxdata', type=int, default=120)
parser.add_argument('--maxans', type=int, default=30)
parser.add_argument('--vocab', type=int, default=21)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--drop', type=float, default=0.1)
parser.add_argument('--dmodel', type=int, default=256)
parser.add_argument('--num_layer', type=int, default=3)
parser.add_argument('--head', type=int, default=4)
parser.add_argument('--num_range', type=int, default=11)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--sft', action='store_true', default=False)
parser.add_argument('--fp16', action='store_true', default=False)
import torch.backends.cudnn as cudnn
args = parser.parse_args()
# Enable cuDNN benchmarking and deterministic mode
cudnn.benchmark = True
cudnn.deterministic = False

# Set torch.backends memory allocation to reduce memory fragmentation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




import sys
sys.path.append(args.folder)
from dataloader_test import getLoader
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'




main_process = 0
set_seed(2023)
dist.init_process_group(backend='nccl')

local_rank = int(os.environ["LOCAL_RANK"])
if dist.get_rank() == 0:
    wandb.init(
        project=f"{args.folder}_eval",  # Change this to your project name
        config=vars(args),
        name=f"{args.folder}_{args.name}"
    )
torch.cuda.set_device(local_rank)
dist.barrier()
model = GPT(args).cuda()


import time
criterion = nn.CrossEntropyLoss(ignore_index=0)
for i in range(1, 11):
    if args.model_path:
        
        #epoch_str = f"epoch_{i}" if i < 10 else f"epoch_{i}"
        epoch_str = f"step_ratio_0.{i}" if i < 10 else f"step_ratio_1.0"
        print(f"{os.path.join(args.model_path, epoch_str)}.pt")
        # Create new model instance for each epoch to avoid DDP issues
        model = GPT(args).cuda()
        model.load_state_dict(torch.load(f"{os.path.join(args.model_path, epoch_str)}.pt"), strict=True)
        model = DDP(model, device_ids=[local_rank])
        model_without_ddp = model.module
        test_loader, test_loader_IID1, test_loader_IID2 = getLoader(args)
        from eval_test import evaluate
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            acc, val_loss = evaluate(model_without_ddp, test_loader, args, 0)
            acc_IID1, val_loss_IID1 = evaluate(model_without_ddp, test_loader_IID1, args, 1)
            acc_IID2, val_loss_IID2 = evaluate(model_without_ddp, test_loader_IID2, args, 2)
            if dist.get_rank() == main_process:
                wandb.log({
                    "val_acc_OOD": acc,
                    "val_loss_OOD": val_loss,
                    "val_acc_IID1": acc_IID1,
                    "val_loss_IID1": val_loss_IID1,
                    "val_acc_IID2": acc_IID2,
                    "val_loss_IID2": val_loss_IID2
                }, step=i)  # This will use the epoch number as the x-axis step
    else:
        raise ValueError("model_path is required")

