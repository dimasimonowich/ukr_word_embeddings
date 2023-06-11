import torch
import torch.nn as nn
import math
from config import CONFIG


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = CONFIG["tf"]["embedding_dim"]
        self.max_len = CONFIG["pe"]["max_len"]

        self.dropout = nn.Dropout(CONFIG["pe"]["dropout_prob"])

        pos_encoding = torch.zeros(self.max_len, self.embedding_dim)
        positions_list = torch.arange(0, self.max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0)) / self.embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
