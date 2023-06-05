from torch import nn
from config import CONFIG


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.input_size = CONFIG["model"]["input_size"]
        self.hidden_size = CONFIG["model"]["hidden_size"]

        self.encoder = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x
