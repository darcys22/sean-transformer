import math
import random
import os

import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import requests
from tqdm import tqdm, trange

import model

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Get default arguments
model_args: model.ModelArgs = model.ModelArgs()

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

# Download Tiny Shakespeare dataset
def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("data/tinyshakespeare.txt"):
        data = requests.get(url).text
        with open("data/tinyshakespeare.txt", "w") as f:
            f.write(data)

# Download the dataset
download_tiny_shakespeare()

# Read the dataset
with open("data/tinyshakespeare.txt", "r") as f:
    text = f.read()

# Create dataset and split into train and test
dataset = TinyShakespeareDataset(text, model_args.seq_length)
model_args.vocab_size = len(dataset.chars)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
loaders = {
    'train': DataLoader(train_dataset, batch_size=model_args.BS, shuffle=True),
    'test': DataLoader(test_dataset, batch_size=model_args.BS, shuffle=True),
}

# Get model
seanTransformer = model.Transformer(model_args)
seanTransformer.to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(seanTransformer.parameters(), lr = 5e-5)

num_epochs = 50
train_losses = {}

log_dir = "data"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for epoch in range(num_epochs):
    epoch_losses = list()
    for i, (X, Y) in enumerate(loaders['train']):
        if X.shape[0] != model_args.BS:
            continue
        optim.zero_grad()
        X, Y, = X.to(device), Y.to(device)
        out = seanTransformer(X)
        out = out.transpose(1, 2)  # batch_size x vocab_size
        loss = criterion(out, Y.long())
        loss.backward()
        optim.step()

        epoch_losses.append(loss.detach().item() / X.shape[1])
        if (i+1) % 250 == 0:
            print('Loss: {:.4f}'.format(loss.detach().item()))
            with open(log_file, "a") as f:
                f.write(f"{epoch} {i} val {loss.detach().item():.4f}\n")
    train_losses[epoch] = torch.tensor(epoch_losses).mean()
    print(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')
    checkpoint_path = os.path.join(log_dir, f"model_{epoch:05d}.pt")
    checkpoint = {
        'model': seanTransformer.state_dict(),
        'model_args': model_args,
        'epoch': epoch,
        'val_loss': train_losses[epoch]
    }
    torch.save(checkpoint, checkpoint_path)

