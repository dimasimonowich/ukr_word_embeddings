import torch
import torch.nn as nn
import math
from model.positional_encoding import PositionalEncoding
from config import CONFIG


class TransformerED(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "Transformer"

        self.vocab_size = CONFIG["data"]["vocab_size"]
        self.embedding_dim = CONFIG["tf"]["embedding_dim"]
        self.num_head = CONFIG["tf"]["num_head"]
        self.num_encoder_layers = CONFIG["tf"]["num_encoder_layers"]
        self.num_decoder_layers = CONFIG["tf"]["num_decoder_layers"]
        self.dropout_prob = CONFIG["tf"]["dropout_prob"]

        self.positional_encoder = PositionalEncoding()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_head,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout_prob,
        )
        self.out = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_dim)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        transformer_out = self.transformer(
            src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask
        )
        out = self.out(transformer_out)

        return out

    def encode(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.positional_encoder(src)
        src = src.permute(1, 0, 2)

        src = self.transformer.encoder(src, None)

        return src

    def embed(self, src):
        return self.embedding(src)

    @staticmethod
    def get_tgt_mask(size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask

    @staticmethod
    def create_pad_mask(matrix, pad_token):
        return matrix == pad_token
