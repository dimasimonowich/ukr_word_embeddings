from training import Pipeline
from model import TransformerED
from dataset import UberCorpusDataset
from infographic import Infographic
import json


with open("data/processed/uc_025_w8_v10000/tokenizer.json") as f:
    word_2_idx = json.load(f)

dataset = UberCorpusDataset.from_raw(word_2_idx=word_2_idx)
dataset.save_to_files()

dataset = UberCorpusDataset.from_files()
print(dataset.target.shape)


# dataset = UberCorpusDataset.from_files()
# model, train_losses, val_losses, accuracies = Pipeline(TransformerED()).train(dataset)
#
# Infographic(model, train_losses, val_losses, accuracies).plot()


# import json, torch
#
# vocab_path = "data/processed/uc_025_w8_v10000/tokenizer.json"
# context_path = "data/processed/uc_025_w8_v10000/context.pt"
# target_path = "data/processed/uc_025_w8_v10000/target.pt"
# max_val = 9999
#
#
# with open(vocab_path, "r") as f:
#     word_2_idx = json.load(f)
#     word_2_idx_new = {
#         k: v for k, v in word_2_idx.items() if v < max_val
#     }
#     word_2_idx_new["~"] = max_val
#
# with open(vocab_path, "w") as f:
#     json.dump(word_2_idx_new, f)
#
#
# context_new = torch.LongTensor(torch.load(context_path))
# context_new[context_new > max_val - 1] = max_val
#
# target_new = torch.LongTensor(torch.load(target_path))
# target_new[target_new > max_val - 1] = max_val
#
# torch.save(context_new, context_path)
# torch.save(target_new, target_path)
#
# print(torch.max(context_new))
# print(torch.max(target_new))
#
