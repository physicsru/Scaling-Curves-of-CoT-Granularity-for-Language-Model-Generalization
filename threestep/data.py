import numpy as np
import argparse
import os
from typing import List, Tuple
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor
import math
parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=40)
parser.add_argument('--train_size', type=int, default=1)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--number_range', type=int, default=47)
parser.add_argument('--step_ratio', type=float, default=1)
parser.add_argument('--recap', type=int, default=1)
parser.add_argument('--obstacle', type=int, default=4)
parser.add_argument('--num_processes', type=int, default=1)

args = parser.parse_args()

def init_worker():
    seed = (int(time.time() * 10000) + os.getpid()) % 123456789
    np.random.seed(seed)

def get_seq(length, obstacle = 0):
    seq = np.ones(length, dtype=np.int32)
    if obstacle > 0:
        positions = np.random.choice(length, size=obstacle, replace=False)
        seq[positions] = 0
    return seq

def solve_with_steps_accumulate_recap(lst: List[int], num_range: int = 100000001, step_ratio: float = 1, recap: bool = False) -> Tuple[str, int]:
    length = len(lst)
    steps = []
    lst_str = ' '.join(map(str, lst))
    steps.append(lst_str + f" , {length}")
    
    dp = np.zeros(length + 1, dtype=np.int32)
    dp[0] = 1
    
    for i in range(1, length + 1):
        if lst[i-1] == 0:
            dp[i] = 0
        else:
            for step in range(1, 4):
                prev = i - step
                if prev >= 0:
                    dp[i] += dp[prev]
            dp[i] %= num_range

        if i != length:
            if np.random.random() < step_ratio:
                curr_state = [str(int(dp[j])) for j in range(max(0, i-3), i+1)]
                step = f"{str(i)} , {str(lst[i-1])} , {' '.join(curr_state[:-1])} -> {dp[i]}"
                steps.append(step)
        else:
            curr_state = [str(int(dp[j])) for j in range(max(0, i-3), i+1)]
            step = f"{str(i)} , {str(lst[i-1])} , {' '.join(curr_state[:-1])} -> {dp[i]}"
            steps.append(step)
    
    res = int(dp[length] % num_range)
    steps.append(str(res))
    
    return " <sep> ".join(steps)

def generate_batch_samples(batch_size, step_ratio=1, recap=False):
    return [solve_with_steps_accumulate_recap(get_seq(args.length, args.obstacle), 
                                            args.number_range, step_ratio, recap) 
            for _ in range(batch_size)]

def generate_set(size, existing_set=None, step_ratio=1, recap=False):
    data_set = []
    
    # Calculate optimal chunk size based on CPU count and data size
    chunk_size = max(1000, size // (os.cpu_count() * 4))
    chunks = math.ceil(size / chunk_size)
    
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        for i in range(chunks):
            batch_size = min(chunk_size, size - i * chunk_size)
            if batch_size <= 0:
                break
            futures.append(executor.submit(generate_batch_samples, batch_size, step_ratio, recap))
            
        for i, future in enumerate(futures):
            data_set.extend(future.result())
            #print(f"Generated {len(data_set)}/{size} samples")
            
    return data_set

train_set = generate_set(args.train_size, step_ratio=args.step_ratio, recap=args.recap)
test_set = generate_set(args.test_size, step_ratio=args.step_ratio, recap=args.recap)

os.makedirs(args.file, exist_ok=True)
chain = os.path.join(args.file, "chain")
os.makedirs(chain, exist_ok=True)

def write_chain(file_path, data):
    n_max_chain = 0
    with open(file_path, 'w') as f:
        for lst in data:
            n_max_chain = max(n_max_chain, len(lst.split(" ")))
            f.write(lst + "\n")
    return n_max_chain

train_n_max_chain = write_chain(os.path.join(chain, "train_data.txt"), train_set)
test_n_max_chain = write_chain(os.path.join(chain, "test_data.txt"), test_set)

print(f"max direct len: {args.length + 2}")
print(f"max train cot len: {train_n_max_chain}")
print(f"max test cot len: {test_n_max_chain}")
