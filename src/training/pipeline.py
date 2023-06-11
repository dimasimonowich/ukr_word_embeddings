import torch
from config import CONFIG
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class Pipeline:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["training"]["lr"],
        )

        self.num_epochs = CONFIG["training"]["num_epochs"]
        self.vocab_size = CONFIG["data"]["vocab_size"]

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
            val_loss, accuracy = self._validation_loop(val_loader, dataset.idx_2_word)
            train_loss = self._training_loop(train_loader, dataset.idx_2_word)
            train_losses.append(train_loss)

            if (epoch + 1) % validate_on_epoch == 0:
                val_loss, accuracy = self._validation_loop(val_loader, dataset.idx_2_word)
                val_losses.append(val_loss)
                accuracies.append(accuracy)

                print(f'Epoch [{epoch + 1}/{self.num_epochs}]:'
                      f'Train Loss: {train_loss:.5f}; '
                      f'Validation Loss: {val_loss:.5f}; '
                      f'Validation Accuracy: {accuracy * 100:.2f}%; ')

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
            context_batch = context_batch.to(self.device).transpose(0, 1)
            target_batch = target_batch.to(self.device).transpose(0, 1).contiguous()

            self.optimizer.zero_grad()

            src_mask = self.model.generate_square_subsequent_mask(len(context_batch)).to(self.device)
            output = self.model(context_batch, src_mask)

            loss = self.criterion(output.view(-1, self.vocab_size), target_batch.view(-1))
            loss.backward()
            self.optimizer.step()

            loop_losses.append(loss.detach().cpu())

        return np.mean(loop_losses)/len(train_loader)

    def _validation_loop(self, val_loader, idx_2_word):
        self.model.eval()
        loop_losses = []
        correct = 0
        total = 0

        for context_batch, target_batch in tqdm(val_loader):
            context_batch = context_batch.to(self.device).transpose(0, 1)
            target_batch = target_batch.to(self.device).transpose(0, 1).contiguous()

            src_mask = self.model.generate_square_subsequent_mask(len(context_batch)).to(self.device)
            output = self.model(context_batch, src_mask)

            loss = self.criterion(output.view(-1, self.vocab_size), target_batch.view(-1))

            _, predicted = torch.max(output.data, 2)
            total += target_batch.shape[0] * target_batch.shape[1]
            correct += (predicted == target_batch).sum().sum().item()
            loop_losses.append(loss.detach().cpu())

        p = predicted.tolist()
        tb = target_batch.tolist()
        print(np.array([[idx_2_word[item] for item in row] for row in p[:2]], dtype=object))
        print(np.array([[idx_2_word[item] for item in row] for row in tb[:2]], dtype=object))

        return np.mean(loop_losses)/len(val_loader), correct/total
