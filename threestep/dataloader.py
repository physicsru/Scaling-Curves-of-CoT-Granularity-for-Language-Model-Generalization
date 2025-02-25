import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
from functools import lru_cache
import concurrent.futures  # Added import for parallel processing
import os
class MyDataSet(Data.Dataset):
    def __init__(self, args, control):
        self.args = args
        self.dictionary = self._create_dictionary()
        self.X, self.Y, self.Z = self._load_and_process_data(control)
        
    def _create_dictionary(self):
        dictionary = {"<pad>": 0, "<sep>": 1, "<eos>": 2, ",": 3, "->" : 4}
        for i in range(self.args.num_range):
            dictionary[str(i)] = i + 5
        return dictionary
    
    @lru_cache(maxsize=None)
    def _load_file(self, file_path):
        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
            
    def _load_and_process_data(self, control):
        #file_name = self.args.file + ("/chain/" if self.args.chain else "/decoder/")
        file_name = self.args.file + "/chain/"
        file_map = {
            0: "train_data.txt",
            1: "test_data_OOD.txt",
            2: "test_data_IID1.txt",
            3: "test_data_IID2.txt"
        }
        
        data = self._load_file(f"{file_name}{file_map[control]}")
        if self.args.debug:
            data = data[:100]
            
        # Process all data at once using vectorized operations
        X = self._tokenize_batch(data, f"{file_name}{file_map[control]}")
        Y = self._get_Y_batch(X)
        X = X[:, :-1]  # Remove last token for input
        Z = torch.argmax(torch.where(X == self.dictionary['<sep>'], 1, 0), dim=1)
        
        return X, Y, Z
        
    def _tokenize_batch(self, sentences, file_name):
        max_len = self.args.maxlen
        tokens_list = []
        
        def tokenize(sentence):
            tokens = sentence.split()
            if len(tokens) + 1 > max_len:  # +1 for <eos>
                raise RuntimeError("Sequence too long")
            try:
                arr = [self.dictionary[s] for s in tokens] + [2]  # 2 is <eos>
                padding = [0] * (max_len - len(arr))
                arr.extend(padding)
                return arr
            except KeyError as e:
                # Print the token that caused the KeyError
                for token in tokens:
                    if token not in self.dictionary:
                        print(f"KeyError: Token '{token}' not found in dictionary")
                raise RuntimeError(f"KeyError: Token not in dictionary: {str(e)}")
        
        # Use CPU count to determine optimal number of workers
        num_workers = min(len(sentences), os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(tokenize, sentence) for sentence in sentences]
            for future in concurrent.futures.as_completed(futures):
                tokens_list.append(future.result())
        
        if not tokens_list:
            raise RuntimeError(f"{file_name}\nNo valid sequences found in the dataset")
            
        return torch.tensor(tokens_list, dtype=torch.int32)
        
    def _get_Y_batch(self, X):
        Y = X[:, 1:].clone()
        batch_size = Y.shape[0]
        
        # Find positions of <sep> and <eos> tokens
        sep_pos = torch.argmax(torch.where(Y == self.dictionary['<sep>'], 1, 0), dim=1)
        eos_pos = torch.argmax(torch.where(Y == self.dictionary['<eos>'], 1, 0), dim=1)
        
        
        if self.args.sft:
            # Zero out positions before <sep> and after <eos>
            for i in range(batch_size):
                Y[i, :sep_pos[i] + 1] = 0
                Y[i, eos_pos[i] + 1:] = 0
        else:
            for i in range(batch_size):
                Y[i, eos_pos[i] + 1:] = 0
            
        return Y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

def getLoader(args):
    number = 4
    datasets = [MyDataSet(args, i) for i in range(number)]
    samplers = [torch.utils.data.distributed.DistributedSampler(datasets[i]) for i in range(number)]
    
    # Create shared memory tensors for faster data loading
    for dataset in datasets:
        dataset.X = dataset.X.share_memory_()
        dataset.Y = dataset.Y.share_memory_()
        dataset.Z = dataset.Z.share_memory_()
    
    dataloaders = [DataLoader(
        datasets[i],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,  # Increased workers for large dataset
        prefetch_factor=3,  # Increased prefetch for more buffering
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=samplers[i],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    ) for i in range(number)]
    
    # Enable non-blocking data transfers
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    return dataloaders[0], dataloaders[1], dataloaders[2], dataloaders[3]