from data.data import DATA
from data_processing import WordIndexing, Padding, FormCBOWPairs, OneHotEncoding
from training.train import TrainingLoop
from model.nlu import NLUGeneration
from dataset.wiki import WikiDataset
import numpy as np
import torch
from config import CONFIG
import os
from matplotlib import pyplot as plt


model = TrainingLoop(NLUGeneration()).run(WikiDataset())




# wi = WordIndexing()
# padding = Padding()
# cbow = FormCBOWPairs()
#
#
# data = wi(DATA)
# data = padding(data)
# data = cbow(data)
#
# oh_encoding = OneHotEncoding(wi.idx_vocabulary)
# data = oh_encoding(data)
#
# model = TrainingLoop(Embedding(len(wi.idx_vocabulary))).run(data)


###### TESTING #####

# words_idxes = list(range(1, 20))
# words = [wi.idx_2_word[word_idx] for word_idx in words_idxes]
# oh_encoded_words = oh_encoding(np.array(words_idxes))
#
# model = Embedding(len(wi.idx_vocabulary))
# model.load_state_dict(torch.load(os.path.join(CONFIG["training"]["saves_folder"], "model.pt")))
#
# results = {}
# word_embeddings = []
# for i, oh_encoded_word in enumerate(oh_encoded_words):
#     output = model(torch.from_numpy(oh_encoded_word).float())
#     output = output.detach().cpu().numpy()
#
#     output_word = wi.idx_2_word[output.argmax() + 1]
#     results[words[i]] = output_word
#
#     word_embedding = model.encoder(torch.from_numpy(oh_encoded_word).float())
#     word_embeddings.append(word_embedding.detach().cpu().numpy())
#
# word_embeddings = np.array(word_embeddings)
# print(word_embeddings)
#
# fig, ax = plt.subplots()
# ax.scatter(word_embeddings[:, 0], word_embeddings[:, 1])
#
# for i, word in enumerate(words):
#     ax.annotate(word, (word_embeddings[i, 0], word_embeddings[i, 1]))
#
# plt.show()
