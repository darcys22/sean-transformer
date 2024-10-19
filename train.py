import math
import random
import os
import wandb
import logging

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
import karpathy_model
from fineweb import load_data

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logger.info(f"using device: {device}")

# Get default arguments
model_args: model.ModelArgs = model.ModelArgs()
data = load_data(model_args)

# Get model
# seanTransformer = model.Transformer(model_args)
seanTransformer = karpathy_model.GPT(karpathy_model.GPTConfig())
seanTransformer.to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(seanTransformer.parameters(), lr=model_args.learning_rate)

num_epochs = 2
train_losses = {}

wandb.init(
    # set the wandb project where this run will be logged
    project="sean-transformer",

    # track hyperparameters and run metadata
    config=model_args
)

log_dir = "data"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

step = 0
loss_accum = 0
for epoch in range(num_epochs):
    epoch_losses = list()
    optim.zero_grad()  # Reset gradients at the beginning of each epoch
    for i, (X, Y) in enumerate(data.loaders['train']):
        if X.shape[0] != model_args.batch_size:
            continue

        X, Y = X.to(device), Y.to(device)

        # Forward pass
        out = seanTransformer(X)
        out = out.transpose(1, 2)  # Adjust for loss calculation
        loss = criterion(out, Y.long())

        # Scale loss by accumulation steps
        loss = loss / model_args.gradient_accumulation
        loss_accum += loss.detach()

        # Backward pass (accumulate gradients)
        loss.backward()

        # Perform optimization step only after accumulating gradients for `model_args.gradient_accumulation` mini-batches
        if (i + 1) % model_args.gradient_accumulation != 0:
            continue

        step = step + 1
        optim.step()
        optim.zero_grad()  # Reset gradients after each optimizer step

        # Track the loss for logging (multiply by accumulation steps to reflect the correct loss)
        epoch_losses.append(loss.item())

        norm = torch.nn.utils.clip_grad_norm_(seanTransformer.parameters(), 1.0)

        logger.info('Step: {:05d} | Loss: {:.4f} | Norm: {:.4f} '.format(step, loss_accum.item(), norm))
        wandb.log({"loss": loss_accum.item(), "norm": norm})
        with open(log_file, "a") as f:
            f.write(f"{epoch} {i} train {loss_accum.item()}\n")

        loss_accum = 0

        # Evaluate Validation loss
        # if (i + 1) % 250 == 0:

    train_losses[epoch] = torch.tensor(epoch_losses).mean()
    logger.info(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')

    # Save model checkpoint
    checkpoint_path = os.path.join(log_dir, f"model_{epoch:05d}.pt")
    checkpoint = {
        'model': seanTransformer.state_dict(),
        'model_args': model_args,
        'epoch': epoch,
        'val_loss': train_losses[epoch]
    }
    torch.save(checkpoint, checkpoint_path)

wandb.finish()

