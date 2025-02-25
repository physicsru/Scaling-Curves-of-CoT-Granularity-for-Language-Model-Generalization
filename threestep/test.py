import numpy as np
from typing import List, Tuple

def get_seq(length, obstacle):
    # Generate sequence of length with 0s and 1s
    seq = np.ones(length, dtype=np.int32)
    
    # Randomly insert obstacle 0s
    if obstacle > 0:
        positions = np.random.choice(length, size=obstacle, replace=False)
        seq[positions] = 0
        
    return seq

def solve_with_steps_accumulate_recap(lst: List[int], num_range: int = 100000001, step_ratio: float = 1, recap: bool = False) -> Tuple[str, int]:
    """Solve how many ways to reach the end of sequence with 1-3 steps, avoiding obstacles"""
    length = len(lst)
    steps = []
    
    # Add initial sequence
    lst_str = ' '.join(map(str, lst))
    steps.append(lst_str + f" , {length}")
    
    # dp[i] represents number of ways to reach position i
    dp = np.zeros(length + 1, dtype=np.int32)
    dp[0] = 1  # Base case - one way to start at position 0
    
    # For each position
    for i in range(1, length + 1):
        
        if lst[i-1] == 0:
            dp[i] = 0
        else:
            # Try steps of size 1, 2, 3
            for step in range(1, 4):
                prev = i - step
                # Check if previous position is valid and reachable
                if prev >= 0:
                    dp[i] += dp[prev]
            dp[i] %= num_range
        # Add explanation step with probability step_ratio
        if np.random.random() < step_ratio:
            curr_state = []
            for j in range(i+1):
                curr_state.append(str(int(dp[j])))
            step = f"{str(lst[i-1])} , {' '.join(curr_state[-4:-1])} -> {int(dp[i])} , {i}/{length}"
            steps.append(step)
    
    res = int(dp[length] % num_range)
    steps.append(str(res))
    
    return steps, res

lst = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
steps, res = solve_with_steps_accumulate_recap(lst)
print("\n".join(steps))
print(res)
    

