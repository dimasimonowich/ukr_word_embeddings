from matplotlib import pyplot as plt


class Infographic:
    def __init__(self, model, train_losses, val_losses, accuracies):
        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.accuracies = accuracies

    def plot(self):
        plt.plot(self.train_losses)
        plt.plot(self.val_losses)
        plt.show()

        plt.plot(self.accuracies)
        plt.show()

    def perform(self):
        pass
