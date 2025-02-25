import numpy as np
import argparse
import os
from typing import List, Tuple
import multiprocessing as mp
from functools import partial
import time
parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=16)
parser.add_argument('--train_size', type=int, default=1)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--number_range', type=int, default=50)
parser.add_argument('--step_ratio', type=float, default=1)
parser.add_argument('--recap', type=int, default=1)
parser.add_argument('--num_processes', type=int, default=1)

args = parser.parse_args()

# Move seed initialization to worker processes
def init_worker():
    # Use process ID and current time for unique seeds across workers
    seed = (int(time.time() * 10000) + os.getpid()) % 123456789
    np.random.seed(seed)

num_start = args.length + 1

def get_seq(length):
    sub_seq = np.random.randint(3) + 1
    total = np.random.randint(length - 2) + 3
    if sub_seq == 1:
        increasing = [total]
    elif sub_seq == 2:
        tmp = np.random.randint(total // 2 + 1) + 1
        increasing = [tmp, total - tmp]
    else:
        tmp1 = np.random.randint(total // 3 + 1) + 1
        tmp2 = np.random.randint((total - tmp1) // 2 + 1) + 1
        increasing = [tmp1, tmp2, total - tmp1 - tmp2]

    chosen = []
    for i in range(sub_seq):
        size = min(increasing[i], args.number_range - num_start)
        vals = np.random.choice(args.number_range - num_start, size=size, replace=False) + num_start
        chosen.append(np.sort(vals))

    numbers = np.concatenate(chosen)
    places = np.sort(np.random.choice(length, total, replace=False))
    seq = np.random.randint(num_start, high=args.number_range, size=length)
    seq[places] = numbers

    return seq

def solve(lst: List[int], step_ratio: float = 1, recap: bool = False) -> Tuple[str, int]:
    length = len(lst)
    cot = np.ones(length, dtype=np.int32)
    res = 0
    steps = []
    steps.append(' '.join(map(str, lst)))
    for l in range(length):
        for i in range(l):
            if lst[l] > lst[i]:
                cot[l] = max(cot[i] + 1, cot[l])
                res = max(res, cot[l])

    
    steps.append(' '.join(map(str, cot)))
    steps.append(str(res))
    return steps, res

def solve_with_steps_accumulate_recap(lst: List[int], step_ratio: float = 1, recap: bool = False) -> Tuple[str, int]:
    """Solve LIS with step by step explanation and optional recap of remaining sequence"""
    length = len(lst)
    dp = np.ones(length, dtype=np.int32)
    steps = []
    lst_str = ' '.join(map(str, lst))
    steps.append(lst_str)
    # DP state
    dp_state = []
    # For each position i
    for i in range(length):
        # Current sequence state
        seq_state = []
        # LIS comparisons
        lis_comp = []
        
        
        # Build states for position i
        for j in range(i+1):
            seq_state.append(str(lst[j]))
            if j < i and lst[i] > lst[j]:
                lis_comp.append('1')
            else:
                lis_comp.append('0')
            #dp_state.append(str(int(dp[j])))
            
        # Update dp[i] based on previous positions
        if i > 0:
            for j in range(i):
                if lst[i] > lst[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            dp_state.append(str(int(dp[i])))
        else:
            dp_state.append(str(1))
        # Add current step if random < step_ratio
        if (step_ratio > 1e-8):
            if i != length - 1:
                if np.random.random() < step_ratio:
                    step = f"{' '.join(seq_state)} , {' '.join(lis_comp)} : {' '.join(dp_state)} -> {int(np.max(dp[:i + 1]))}"
                    steps.append(step)
            else:
                step = f"{' '.join(seq_state)} , {' '.join(lis_comp)} : {' '.join(dp_state)} -> {int(np.max(dp[:i + 1]))}"
                steps.append(step)
        
    return steps, int(np.max(dp))

def solve_completely(lst):
    lst = np.array(lst)
    length = len(lst)
    dp = np.ones(length, dtype=np.int32)
    pred = np.full(length, -1, dtype=np.int32)
    
    # Store the complete solution process
    solution_steps = []
    
    # Initial state
    initial_state = f"Initial dp array: {dp.tolist()}"
    solution_steps.append(initial_state)
    solution_steps.append("<sep>")
    
    for i in range(length):
        current_step = []
        current_step.append(f"Computing dp[{i}] for element {lst[i]}:")
        
        candidates = []
        for j in range(i):
            if lst[i] > lst[j]:
                candidates.append(f"Found smaller element lst[{j}]={lst[j]}, can extend with dp[{j}]+1={dp[j]+1}")
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    pred[i] = j
        
        if not candidates:
            current_step.append(f"No smaller elements found before index {i}, dp[{i}] stays 1")
        else:
            current_step.append("\n".join(candidates))
            current_step.append(f"Updated dp[{i}] to {dp[i]}")
        
        solution_steps.extend(current_step)
        solution_steps.append("<sep>")
        
    solution_steps.append(f"Final dp array: {dp.tolist()}")
    solution_steps.append(f"Length of longest increasing subsequence: {np.max(dp)}")
    
    return solution_steps, np.max(dp)

def generate_single_sample(step_ratio=1, recap=False):
    lst = get_seq(args.length)
    cot, solution = solve_with_steps_accumulate_recap(lst, step_ratio, recap)
    return " <sep> ".join(cot)

def generate_batch(batch_size, step_ratio=1, recap=False):
    samples = set()
    while len(samples) < batch_size:
        sample = generate_single_sample(step_ratio, recap)
        samples.add(sample)
    return samples

def generate_set(size, existing_set=None, step_ratio=1, recap=False):
    data_set = set() if existing_set is None else set(existing_set)
    remaining = size - len(data_set)
    
    if remaining <= 0:
        return data_set
        
    # Split work among processes
    batch_size = max(1, remaining // args.num_processes)
    num_batches = (remaining + batch_size - 1) // batch_size  # Ceiling division
    
    with mp.Pool(processes=args.num_processes, initializer=init_worker) as pool:
        while len(data_set) < size:
            current_batch_size = min(batch_size, size - len(data_set))
            # Create a list of identical batch sizes for parallel processing
            batch_sizes = [current_batch_size] * min(args.num_processes, num_batches)
            #print("there")
            # Map the generate_batch function directly with the batch sizes
            results = pool.map(partial(generate_batch, step_ratio=step_ratio, recap=recap), batch_sizes)
            #print("here")
            for result in results:
                data_set.update(result)
                if len(data_set) >= size:
                    break
                    
            #if len(data_set) % 100000 == 0:
        print(f"Generated {len(data_set)} samples")
                
    return data_set

train_set = generate_set(args.train_size, step_ratio=args.step_ratio, recap=args.recap)
test_set = generate_set(args.test_size, existing_set=train_set, step_ratio=args.step_ratio, recap=args.recap)

os.makedirs(args.file, exist_ok=True)
chain = os.path.join(args.file, "chain")
os.makedirs(chain, exist_ok=True)

def write_chain(file_path, data):
    n_max_chain = 0
    with open(file_path, 'w') as f:
        for lst in data:
            n_max_chain = max(n_max_chain, len(lst.split(" ")))
            f.writelines(lst + "\n")
    return n_max_chain

train_n_max_chain = write_chain(os.path.join(chain, "train_data.txt"), train_set)
test_n_max_chain = write_chain(os.path.join(chain, "test_data.txt"), test_set)

print(f"max direct len: {args.length + 2}")
print(f"max train cot len: {train_n_max_chain}")
print(f"max test cot len: {test_n_max_chain}")
