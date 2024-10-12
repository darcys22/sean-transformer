import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

@dataclass
class ModelArgs:
    emb_size: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    seq_length: int = 128
    BS: int = 32
    vocab_size: int = 64

class MLP(nn.Module):
    def __init__(self, emb_length):
        super().__init__()
        self.i2h = nn.Linear(emb_length, emb_length * 4)
        self.h2o = nn.Linear(emb_length * 4, emb_length)

    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = self.h2o(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.emb_size = args.emb_size
        self.head_dim = self.emb_size // self.n_heads

        assert self.emb_size % self.n_heads == 0, "Embedding size must be divisible by number of heads"

        self.qkv = nn.Linear(self.emb_size, self.emb_size * 3)
        self.fc_out = nn.Linear(self.emb_size, self.emb_size)
        self.feed_forward = MLP(self.emb_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Compute queries, keys, and values in one go
        qkv = self.qkv(x)  # [batch_size, seq_length, emb_size * 3]
        qkv = qkv.reshape(batch_size, seq_length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_length, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [batch_size, n_heads, seq_length, head_dim]

        # Compute scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        seq_len = attn_weights.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)

        out = attn_probs @ v  # [batch_size, n_heads, seq_length, head_dim]
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.emb_size)

        out = self.fc_out(out)
        out = self.feed_forward(out)
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.emb_size)
        self.positional_encoding = nn.Embedding(args.seq_length, args.emb_size)
        self.layers = nn.ModuleList([MultiHeadAttention(args) for i in range(args.n_layers)])
        self.n_layers = args.n_layers
        self.seq_length = args.seq_length
        self.drop = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(args.emb_size)
        self.ll = nn.Linear(args.emb_size, args.vocab_size)

        self.embedding.weight = self.ll.weight


    def forward(self, x):
        b, t = x.size()
        tok_emb = self.embedding(x)
        assert t <= self.seq_length, f"Cannot forward sequence of length {t}, block size is only {self.seq_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device) # shape (t)
        pos_emb = self.positional_encoding(pos) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for transformer_block in self.layers:
            x = transformer_block(x)
        x = self.ln(x)
        x = self.ll(x)
        return x
