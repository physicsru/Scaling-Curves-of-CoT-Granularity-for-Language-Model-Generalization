import torch
import torch.optim as optim
import torch.nn as nn
from model2 import GPT
import argparse
import wandb
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler, set_seed, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.multiprocessing


parser = argparse.ArgumentParser(description='train')

parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--folder', type=str, default="arithmetic")
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--drop', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--warmup', type=float, default=0.1)
parser.add_argument('--output_dir', type=str, default="./output/log")
parser.add_argument('--maxlen', type=int, default=120)
parser.add_argument('--maxdata', type=int, default=120)
parser.add_argument('--maxans', type=int, default=30)
parser.add_argument('--vocab', type=int, default=21)
parser.add_argument('--write2file', action='store_true', default=False)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--dmodel', type=int, default=256)
parser.add_argument('--num_layer', type=int, default=3)
parser.add_argument('--head', type=int, default=4)
parser.add_argument('--num_range', type=int, default=11)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--chain', action='store_true', default=False)
parser.add_argument('--rpe', action='store_true', default=False)
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--sft', action='store_true', default=False)
args = parser.parse_args()
import sys
sys.path.append(args.folder)
from dataloader import getLoader

print("args.sft = ", args.sft)

main_process = 0
set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
dist.init_process_group(backend='nccl')
if dist.get_rank() == main_process:
    wandb.init(
        project=f"{args.folder}_0120",  # Change this to your project name
        config=vars(args),
        name=f"run_{args.folder}_{args.learning_rate}_{args.num_layer}"
    )

# Enable cuDNN benchmarking and deterministic mode
cudnn.benchmark = True
cudnn.deterministic = False

# Set torch.backends memory allocation to reduce memory fragmentation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_optimizer_scheduler(model, args, loader):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Calculate the total number of training steps
    total_steps = len(loader) * args.epoch
    warmup_steps = int(total_steps * args.warmup)
    print("total_steps = ", total_steps)
    print("warmup_steps = ", warmup_steps)
    print("len(loader) = ", len(loader))
    # Create cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    return optimizer, scheduler

local_rank = int(os.environ["LOCAL_RANK"])
print("local_rank = ", local_rank)
torch.cuda.set_device(local_rank)
dist.barrier()
model = GPT(args).cuda()
if args.model_path:
    model.load_state_dict(torch.load(args.model_path), strict=True)
model = DDP(model, device_ids=[local_rank])
model_without_ddp = model.module

loader, test_loader, test_loader_IID1, test_loader_IID2 = getLoader(args)
optimizer, scheduler = set_optimizer_scheduler(model, args, loader)

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

# Initialize gradient scaler for AMP
print("args.fp16 = ", args.fp16)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
cnt = 0
for epoch in range(args.epoch):
    dist.barrier()
    model.train()
    loader.sampler.set_epoch(epoch)
    pbar = tqdm(loader) if not args.write2file and dist.get_rank() == main_process else loader
    
    for data_iter_step, (input_ids, y, _) in enumerate(pbar):
        inputs, y = input_ids.cuda(), y.long().cuda()
        
        optimizer.zero_grad()
        
        # Implement gradient accumulation
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits = model(inputs)
            # Check for NaN in logits
            if torch.isnan(logits).any():
                print("NaN detected in logits!")
                continue
                
            # Add shape checks
            B, S, V = logits.shape
            if y.shape != (B, S):
                print(f"Shape mismatch: logits {logits.shape}, targets {y.shape}")
                continue
                
            loss = criterion(logits.reshape(-1, V), y.reshape(-1))
            
            # Check for NaN in loss
            if torch.isnan(loss).any():
                print("NaN detected in loss!")
                continue
                
            loss = loss / args.gradient_accumulation_steps
        
        # Gradient scaling
        scaler.scale(loss).backward()
        
        if (data_iter_step + 1) % args.gradient_accumulation_steps == 0:
            # Unscale gradients
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for NaN in gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("NaN detected in gradients!")
                optimizer.zero_grad()
                continue
                
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        # Calculate step ratio (0.0 to 1.0) within the epoch
        step_ratio = data_iter_step / len(loader)
        
        # Log metrics
        if dist.get_rank() == main_process:
            wandb.log({
                'loss': loss.item() * args.gradient_accumulation_steps,
                'epoch': epoch + step_ratio,
                'learning_rate': scheduler.get_last_lr()[0],
                'step_ratio': step_ratio
            })
            
        # Evaluate every 10% of epoch
        if step_ratio >= 0.05 and (data_iter_step % (len(loader) // 10) == 0):
            from eval import evaluate
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
                acc, val_loss = evaluate(model_without_ddp, test_loader, args, 0)
                acc_IID1, val_loss_IID1 = evaluate(model_without_ddp, test_loader_IID1, args, 1)
                acc_IID2, val_loss_IID2 = evaluate(model_without_ddp, test_loader_IID2, args, 2)
                if dist.get_rank() == main_process:
                    print(f"Step ratio {step_ratio:.1f} - test acc:{acc}")
                    wandb.log({
                        "step_ratio": step_ratio,
                        "val_acc": acc, 
                        "val_loss": val_loss,
                        "val_acc_IID1": acc_IID1,
                        "val_loss_IID1": val_loss_IID1,
                        "val_acc_IID2": acc_IID2,
                        "val_loss_IID2": val_loss_IID2
                    })
                    cnt += 1
                    # Save model after evaluation
                    model_save_path = f"{args.output_dir}/step_ratio_{step_ratio:.1f}.pt"
                    torch.save(model_without_ddp.state_dict(), model_save_path)
                    wandb.save(model_save_path)

    # if (epoch + 1) % 1 == 0:
    #     dist.barrier()
    #     if dist.get_rank() == main_process:
    #         # Save model in a background thread
    #         model_save_path = f"{args.output_dir}/epoch_{epoch+1}.pt"
    #         torch.save(model_without_ddp.state_dict(), model_save_path)
    #         wandb.save(model_save_path)
    
    # # Evaluate less frequently to save time
    # if (epoch + 1) % 1 == 0:  # Evaluate every 5 epochs
    #     from eval import evaluate
    #     with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
    #         acc, val_loss = evaluate(model_without_ddp, test_loader, args, 0)
    #         acc_IID1, val_loss_IID1 = evaluate(model_without_ddp, test_loader_IID1, args, 1)
    #         acc_IID2, val_loss_IID2 = evaluate(model_without_ddp, test_loader_IID2, args, 2)
    #         if dist.get_rank() == main_process:
    #             print(f"test acc:{acc}")
    #             wandb.log({"epoch": epoch, "val_acc": acc, "val_loss": val_loss})
    #             wandb.log({"epoch": epoch, "val_acc_IID1": acc_IID1, "val_loss_IID1": val_loss_IID1})
    #             wandb.log({"epoch": epoch, "val_acc_IID2": acc_IID2, "val_loss_IID2": val_loss_IID2})

if dist.get_rank() == main_process:
    wandb.finish()