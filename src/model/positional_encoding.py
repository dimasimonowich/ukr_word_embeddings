import math
import torch
from torch import nn
from config import CONFIG


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        self.dropout_prob = CONFIG["pe"]["dropout"]
        self.max_len = CONFIG["pe"]["max_len"]
        self.embedding_dim = CONFIG["tf"]["embedding_dim"]

        self.dropout = nn.Dropout(p=self.dropout_prob)

        pe = torch.zeros(self.max_len, self.embedding_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
