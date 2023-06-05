import torch
from torch import nn
from config import CONFIG
from matplotlib import pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


class TrainingLoop:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG["training"]["lr"],
            momentum=CONFIG["training"]["momentum"],
            weight_decay=CONFIG["training"]["weight_decay"]
        )

        self.num_epochs = CONFIG["training"]["num_epochs"]

        self.saves_path = os.path.join(CONFIG["training"]["saves_folder"], "model.pt")

    def run(self, dataset):
        batch_size = CONFIG["training"]["batch_size"]
        n_samples = len(dataset)
        train_test_split = CONFIG["data"]["train_size"] / (CONFIG["data"]["test_size"] + CONFIG["data"]["train_size"])
        split_location = int(n_samples * train_test_split)

        train_indices, val_indices = np.arange(split_location), np.arange(split_location, n_samples)
        train_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
        )
        val_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
        )

        self.model.train(True)

        losses = []
        min_loss = None

        for epoch in range(self.num_epochs):
            epoch_losses = []

            for context_batch, target_batch in tqdm(train_dataloader):
                output, _, _ = self.model(context_batch)
                loss = self.criterion(output, target_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 1 == 0:
                mean_epoch_loss = np.mean(epoch_losses)
                losses.append(mean_epoch_loss)
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {mean_epoch_loss:.4f}')

                if min_loss is None or mean_epoch_loss < min_loss:
                    min_loss = mean_epoch_loss
                    torch.save(self.model.state_dict(), self.saves_path)

        plt.plot(losses)
        plt.show()

        self.model.train(False)

        return self.model
