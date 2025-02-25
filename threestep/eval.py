import torch
import torch.distributed as dist
import time
# In eval.py
def create_reverse_dictionary(num_range):        
    dictionary = {"<pad>": 0, "<sep>": 1, "<eos>": 2, ",": 3, "->" : 4}
    for i in range(num_range):
        dictionary[str(i)] = i + 5
    return {v: k for k, v in dictionary.items()}

def evaluate(model, cur_loader, args, control = 0):
    Sum = torch.tensor(0).cuda()
    correct = torch.tensor(0).cuda()
    total_loss = torch.tensor(0.0).cuda()
    rev_dict = create_reverse_dictionary(args.num_range)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    predictions = []
    cur_loader.sampler.set_epoch(0)
    for input_ids, y, _ in cur_loader:
        inputs, y = input_ids.cuda(), y.cuda()
        logits = model(inputs)
        pred = torch.argmax(logits, dim=2)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * inputs.shape[0]
        
        # Calculate accuracy metrics
        Sum += torch.as_tensor(inputs.shape[0]).cuda()
        # truth = torch.where(y > 0, 1, 0)
        # predict = torch.where(pred == y, 1, 0) * truth
        # correct += torch.sum(torch.where(torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0))
        
        # Decode and save predictions
        for i in range(y.shape[0]):
            #print(input_ids[i])
            input_tokens = [rev_dict[idx.item()] for idx in input_ids[i] if idx.item() not in [0]]
            gt_tokens = [rev_dict[idx.item()] for idx in y[i] if idx.item() not in [0]]
            pred_tokens = [rev_dict[idx.item()] for idx in pred[i] if idx.item() not in [0]]
            
            input_str = ' '.join(input_tokens)
            gt_str = ' '.join(gt_tokens)
            pred_str = ' '.join(pred_tokens)
            
            predictions.append(f"Input: {input_str.strip()}\nPrediction: {pred_str.strip()}\nGround Truth: {gt_str.strip()}\n\n")
            
            if args.chain:
                try:
                    if control == 0:
                        gt_split = gt_str.split()
                        gt = " ".join(gt_split[-10:-2])
                        start_pos = pred_str.find(gt)
                        if start_pos != -1:
                            end_pos = start_pos + len(gt)
                            print(f"Found match from position {start_pos} to {end_pos}")
                            correct += torch.tensor(1).cuda()       
                    elif control == 1:
                        gt_split = gt_str.split()
                        gt = " ".join(gt_split[-10:-2])
                        start_pos = pred_str.find(gt)
                        if start_pos != -1:
                            end_pos = start_pos + len(gt)
                            print(f"Found match from position {start_pos} to {end_pos}")
                            correct += torch.tensor(1).cuda()       
                    elif control == 2:
                        gt_split = gt_str.split()
                        gt = " ".join(gt_split[-10:-2])
                        start_pos = pred_str.find(gt)
                        if start_pos != -1:
                            end_pos = start_pos + len(gt)
                            print(f"Found match from position {start_pos} to {end_pos}")
                            correct += torch.tensor(1).cuda()        
                except:
                    print(pred_str)
                    print(gt_str)
                    continue
            else:
                print("no cot mode eval")
                try:
                    pred_digits = [x for x in pred_str.split() if x.isdigit()]
                    if pred_digits:
                        pred_num = pred_digits[-1]
                        gt_num = gt_str.split('<eos>')[0].strip()
                        correct += torch.tensor(1 if pred_num == gt_num else 0).cuda()
                except:
                    print(pred_str)
                    print(gt_str)
                    continue
            
    dist.all_reduce(correct)
    dist.all_reduce(Sum)
    dist.all_reduce(total_loss)
    
    # Save predictions to file (only from main process)
    if dist.get_rank() == 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f'predictions_LIS_{args.jobid}_{control}_{timestamp}.txt', 'w') as f:
            f.writelines(predictions)
    
    avg_loss = total_loss.item() / Sum.item()
    return correct.item() / Sum.item(), avg_loss