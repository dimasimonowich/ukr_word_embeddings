from collections import Counter, defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CONFIG
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')
stop = stopwords.words('english')


class WikiDataset(Dataset):
    def __init__(self):
        self.vocab_size = CONFIG["data"]["vocab_size"]
        self.right_window_size = CONFIG["cbow"]["right_window_size"]
        self.left_window_size = CONFIG["cbow"]["left_window_size"]

        self.text = []
        df = pd.read_csv(CONFIG["data"]["path"])["text"][:CONFIG["data"]["train_size"] + CONFIG["data"]["test_size"]]
        self._preprocess_text(df)

        # full_vocabulary = pd.Series(' '.join(self.text).split()).unique()
        self.word_2_idx = defaultdict(lambda: self.vocab_size - 1)
        most_common_words = {
            x[0]: i
            for i, x in enumerate(
                Counter(self.text).most_common()[:self.vocab_size - 1]
            )
        }
        self.word_2_idx.update(most_common_words)
        self.word_2_idx[CONFIG["data"]["unk_token"]] = self.vocab_size - 1

        self.idx_2_word = {v: k for k, v in self.word_2_idx.items()}
        self.vocabulary = list(self.word_2_idx.keys())

    def _preprocess_text(self, df):
        df = df.str.lower()
        df = df.str.replace('[^a-z\s]', '', regex=True)
        df = df.apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop])
        )
        self.text = ' '.join(df).split()

    def __len__(self):
        return len(self.text) - self.left_window_size - self.right_window_size - 1

    def __getitem__(self, idx):
        # len(self.text) = 100
        # idx: 0...95

        # idx = 0
        # context = [0, 1, 3, 4]
        # target = 2

        # idx = 95
        # context = [95, 96, 98, 99]
        # target = 97

        target_location = idx + self.left_window_size
        left_context_words = self.text[idx: target_location]
        right_context_words = self.text[target_location + 1: target_location + 1 + self.right_window_size]

        context = torch.LongTensor(
            [
                self.word_2_idx[c]
                for c in left_context_words + right_context_words
            ]
        )
        target = self.word_2_idx[self.text[target_location]]

        return context, target
