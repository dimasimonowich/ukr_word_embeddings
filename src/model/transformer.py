import math
import torch
from torch import nn
from config import CONFIG
from model.positional_encoding import PositionalEncoding


class CBOWTransformer(nn.Module):
    def __init__(self):
        super(CBOWTransformer, self).__init__()

        self.vocab_size = CONFIG["data"]["vocab_size"]
        self.embedding_dim = CONFIG["tf"]["embedding_dim"]
        self.dropout = CONFIG["tf"]["dropout"]
        self.num_head = CONFIG["tf"]["num_head"]
        self.hidden_dim = CONFIG["tf"]["hidden_dim"]
        self.num_layers = CONFIG["tf"]["num_layers"]

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embedding_dim, self.num_head, self.hidden_dim, self.dropout),
            self.num_layers
        )
        self.pos_encoder = PositionalEncoding()

        self.decoder = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    @staticmethod
    def generate_square_subsequent_mask(size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class CBOWTransformerEncoder(CBOWTransformer):
    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        output = self.transformer_encoder(src, src_mask)
        return output
