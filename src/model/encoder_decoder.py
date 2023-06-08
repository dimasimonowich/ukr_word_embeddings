from torch.nn import Embedding, Linear, LSTM, Module, ReLU
from config import CONFIG


class EncoderDecoder(Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = CONFIG["data"]["vocab_size"]
        self.embedding_dim = CONFIG["model"]["embedding_dim"]
        self.hidden_dim = CONFIG["model"]["hidden_dim"]
        self.num_layers = CONFIG["model"]["num_layers"]
        self.fc_dim = CONFIG["model"]["fc_dim"]
        self.context_length = CONFIG["cbow"]["right_window_size"] + CONFIG["cbow"]["left_window_size"]

        self.embedding = Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=self.vocab_size - 1,
            norm_type=2,
            max_norm=2,
        )

        self.lstm = LSTM(
            self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers
        )

        self.fc1 = Linear(self.hidden_dim, self.fc_dim)
        self.fc2 = Linear(self.fc_dim, self.vocab_size)

        self.relu = ReLU()

    def forward(self, x, h=None, c=None):
        x = self.embedding(x)

        if h is not None and c is not None:
            _, (h, c) = self.lstm(x, (h, c))
        else:
            _, (h, c) = self.lstm(x)  # (n_layers, n_samples, hidden_dim)
        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)

        x = self.relu(self.fc1(h_mean))
        x = self.fc2(x)

        return x, h, c
