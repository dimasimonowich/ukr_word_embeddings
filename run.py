from training import Pipeline
from model import CBOWTransformer
from dataset import BrukDataset, UberCorpusDataset
from infographic import Infographic

# dataset = BrukDataset.from_files()
# model, train_losses, val_losses, accuracies = Pipeline(CBOWTransformer()).train(dataset)
#
# Infographic(model, train_losses, val_losses, accuracies).plot()


dataset = UberCorpusDataset.from_raw()
dataset.save_to_files()
dataset = UberCorpusDataset.from_files()
print(dataset.target.shape)
