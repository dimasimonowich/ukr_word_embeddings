import torch
from torch import nn
from config import CONFIG
from matplotlib.pyplot import plt


class TrainingLoop:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG["training"]["lr"],
            momentum=CONFIG["training"]["momentum"],
            weight_decay=CONFIG["training"]["weight_decay"]
        )
        self.num_epochs = CONFIG["training"]["num_epochs"]

    def run(self, data):
        data = data.to(self.device)
        self.model.train(True)

        losses = []

        for epoch in range(self.num_epochs):
            for i, pair in enumerate(data):
                target, context = pair

                # Forward pass
                output = self.model(context)
                loss = self.criterion(output, target)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {losses[-1].item():.4f}')

        plt.plot(losses)