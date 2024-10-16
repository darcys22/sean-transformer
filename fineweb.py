import os
import multiprocessing as mp
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import tiktoken

# Function to tokenize a document
def tokenize_document(doc):
    global enc, eot
    tokens = [eot]  # <|endoftext|> token
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

# Function to initialize worker processes
def init_worker():
    global enc, eot
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']  # end of text token

# Function to load tokens from a file
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

# Function to check shard integrity
def check_shard_integrity(filename):
    try:
        tokens = np.load(filename)
        return tokens.size > 0  # Ensure the shard contains tokens
    except Exception as e:
        print(f"Failed to load shard {filename}: {e}")
        return False

# Class to handle data loading for FineWeb dataset
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        # Get the shard filenames
        self.shards = self.get_shards(data_root, split)
        self.reset()

    def get_shards(self, data_root, split):
        shards = [s for s in sorted(os.listdir(data_root)) if split in s and s.endswith('.npy')]
        assert len(shards) > 0, f"No shards found for split '{split}' in {data_root}."
        return [os.path.join(data_root, s) for s in shards]

    def reset(self):
        # Initialize state at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def __iter__(self):
        """Return the iterator object (self) when used in an iteration context."""
        self.reset()  # Reset iterator each time it's called
        return self

    def __next__(self):
        """Return the next batch of data or raise StopIteration."""
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            # If loading the next batch would go out of bounds, load the next shard
            self.current_shard += 1
            if self.current_shard >= len(self.shards):
                raise StopIteration  # No more shards, stop iteration

            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank  # Reset position for new shard

        # Get the next batch
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]

        if len(buf) <= 1:
            raise StopIteration  # No more data in the current shard

        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        self.current_position += B * T * self.num_processes

        return x, y

# Dataset loader wrapper class for FineWeb
class FineWebLoaderWrapper:
    def __init__(self, model_args, process_rank=0, num_processes=1):
        self.model_args = model_args
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Set dataset-specific configurations
        self.local_dir = "data/edu_fineweb10B"
        self.remote_name = "sample-10BT"
        self.shard_size = int(1e8)  # 100M tokens per shard
        self.DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), self.local_dir)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)

        # Load the dataset and shards
        self.load_fineweb_dataset()

        self.model_args.vocab_size = enc.n_vocab  # Set vocab size based on tokenizer


        # Initialize DataLoaderLite for train and val splits
        B, T = self.model_args.batch_size, self.model_args.seq_length
        self.loaders = {
            'train': DataLoaderLite(B=B, T=T, process_rank=self.process_rank,
                                    num_processes=self.num_processes, split="train", data_root=self.DATA_CACHE_DIR),
            'val': DataLoaderLite(B=B, T=T, process_rank=self.process_rank,
                                  num_processes=self.num_processes, split="val", data_root=self.DATA_CACHE_DIR)
        }

    def load_fineweb_dataset(self):
        # Check if shards already exist and validate their integrity
        train_shards = [f for f in os.listdir(self.DATA_CACHE_DIR) if 'train' in f and f.endswith('.npy')]
        val_shards = [f for f in os.listdir(self.DATA_CACHE_DIR) if 'val' in f and f.endswith('.npy')]

        # Initialize tokenizer
        init_worker()  # Initialize enc and eot

        if train_shards and val_shards:
            print("Checking shard integrity...")
            if all(check_shard_integrity(os.path.join(self.DATA_CACHE_DIR, f)) for f in train_shards + val_shards):
                print("All shards are valid. Skipping dataset processing.")
                return
            else:
                print("Invalid shards found. Reprocessing dataset.")

        # Download the dataset from Hugging Face and cache it locally
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=self.remote_name, split="train", cache_dir=self.DATA_CACHE_DIR)


        # Tokenize documents and create shards
        nprocs = max(1, os.cpu_count() // 2)
        with mp.Pool(nprocs, initializer=init_worker) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None

            for tokens in pool.imap(tokenize_document, fw, chunksize=16):
                # Check if there is enough space in the current shard for the new tokens
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)

                    if progress_bar is None:
                        progress_bar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    # Determine split based on shard index
                    split = 'val' if shard_index == 0 else 'train'
                    filename = os.path.join(self.DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")

                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                    np.save(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder

            # Write any remaining tokens as the last shard
            if token_count != 0:
                split = 'val' if shard_index == 0 else 'train'
                filename = os.path.join(self.DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
                np.save(filename, all_tokens_np[:token_count])

# Function to load the FineWeb dataset and return data loaders
def load_data(model_args, process_rank=0, num_processes=1):
    return FineWebLoaderWrapper(model_args, process_rank, num_processes)

# Ensure this code is only executed when running as the main module
if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class ModelArgs:
        emb_size: int = 1024
        n_layers: int = 8
        n_heads: int = 8
        seq_length: int = 128
        batch_size: int = 32
        vocab_size: int = 64

    model_args = ModelArgs()
    data = load_data(model_args)

