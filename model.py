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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_length, n_heads):
        super().__init__()
        self.head_size = emb_length // n_heads
        self.key = nn.Linear(self.head_size, self.head_size)
        self.query = nn.Linear(self.head_size, self.head_size)
        self.value = nn.Linear(self.head_size, self.head_size)
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(self.head_size)
        self.ln2 = nn.LayerNorm(self.head_size)

    def forward(self, x):
        xq = self.query(x)
        xk = self.key(x)
        xv = self.value(x)
        att = (xq @ xk.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_size))
        # TODO sean mask here
        x = F.softmax(att, dim=-1)
        x = x @ xv
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.emb_size = args.emb_size
        self.heads = nn.ModuleList([ScaledDotProductAttention(args.emb_size, args.n_heads) for i in range(args.n_heads)])
        self.feed_forward = MLP(args.emb_size)

    def forward(self, x):
        x = torch.split(x, self.emb_size // self.n_heads, 2)
        attention_heads = []
        for i in range(self.n_heads):
            head_x = self.heads[i](x[i])
            attention_heads.append(head_x)
        x = torch.cat(attention_heads, 2)
        x = self.feed_forward(x)
        return x

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
        x = F.softmax(x, dim=-1)
        return x
