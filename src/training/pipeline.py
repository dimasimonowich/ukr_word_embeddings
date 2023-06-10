import torch
from torch import nn
from config import CONFIG
from matplotlib import pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class Pipeline:
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

    def train(self, dataset):
        batch_size = CONFIG["training"]["batch_size"]
        validate_on_epoch = CONFIG["training"]["validate_on_epoch"]

        train_ratio = CONFIG["data"]["train_size"] / (CONFIG["data"]["test_size"] + CONFIG["data"]["train_size"])
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.train(True)

        train_losses, val_losses, accuracies = [], [], []
        min_val_loss = None

        for epoch in range(self.num_epochs):
            train_loss = self._training_loop(train_loader)
            train_losses.append(train_loss)

            if (epoch + 1) % validate_on_epoch == 0:
                val_loss, accuracy = self._validation_loop(val_loader)
                val_losses.append(val_loss)
                accuracies.append(accuracy)

                print(f'Epoch [{epoch + 1}/{self.num_epochs}]:'
                      f'Train Loss: {train_loss:.4f}; '
                      f'Validation Loss: {val_loss:.4f}; '
                      f'Validation Accuracy: {accuracy:.4f}; ')

                if min_val_loss is None:
                    min_val_loss = val_loss

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.saves_path)

        return self.model, train_losses, val_losses, accuracies

    def _training_loop(self, train_loader):
        self.model.train()
        loop_losses = []

        for context_batch, target_batch in tqdm(train_loader):
            context_batch, target_batch = context_batch.to(self.device), target_batch.to(self.device)

            self.optimizer.zero_grad()

            output, _, _ = self.model(context_batch)
            loss = self.criterion(output, target_batch)
            loss.backward()
            self.optimizer.step()

            loop_losses.append(loss.detach().cpu())

        return np.mean(loop_losses)/len(train_loader)

    def _validation_loop(self, val_loader):
        self.model.eval()
        loop_losses = []
        correct = 0
        total = 0

        for context_batch, target_batch in tqdm(val_loader):
            context_batch, target_batch = context_batch.to(self.device), target_batch.to(self.device)

            output, _, _ = self.model(context_batch)
            loss = self.criterion(output, target_batch)

            _, predicted = torch.max(output.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum().item()

            loop_losses.append(loss.detach().cpu())

        return np.mean(loop_losses)/len(val_loader), correct/total
