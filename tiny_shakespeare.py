import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Custom dataset class for Tiny Shakespeare
class TinyShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data_size = len(self.text)

    def __len__(self):
        return max(0, self.data_size - self.seq_length)

    def __getitem__(self, index):
        x = [self.char_to_idx[c] for c in self.text[index:index+self.seq_length]]
        y = [self.char_to_idx[c] for c in self.text[index+1:index+self.seq_length+1]]
        return torch.tensor(x), torch.tensor(y)

# Function to download Tiny Shakespeare dataset
def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("data/tinyshakespeare.txt"):
        os.makedirs("data", exist_ok=True)
        data = requests.get(url).text
        with open("data/tinyshakespeare.txt", "w") as f:
            f.write(data)

# The dataset loader class
class DataLoaderWrapper:
    def __init__(self, model_args):
        self.model_args = model_args
        self.load_tiny_shakespeare_dataset()

    def load_tiny_shakespeare_dataset(self):
        # Download dataset if not available
        download_tiny_shakespeare()

        # Read dataset
        with open("data/tinyshakespeare.txt", "r") as f:
            text = f.read()

        # Create TinyShakespeareDataset instance
        dataset = TinyShakespeareDataset(text, self.model_args.seq_length)

        # Update the vocab_size in model_args
        self.model_args.vocab_size = len(dataset.chars)

        # Split dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create data loaders
        self.loaders = {
            'train': DataLoader(train_dataset, batch_size=self.model_args.batch_size, shuffle=True),
            'test': DataLoader(test_dataset, batch_size=self.model_args.batch_size, shuffle=True),
        }
        # Other useful properties
        self.vocab_size = self.model_args.vocab_size
        self.seq_length = self.model_args.seq_length

# Function to load data based on dataset name
def load_data(model_args):
    return DataLoaderWrapper(model_args)

