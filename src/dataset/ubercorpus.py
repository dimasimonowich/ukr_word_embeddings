from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CONFIG
import re
from tqdm import tqdm
import json
import numpy as np


class UberCorpusDataset(Dataset):
    vocab_size = CONFIG["data"]["vocab_size"]
    left_window_size = CONFIG["data"]["window_size"]

    def __init__(
        self,
        context,
        target,
        word_2_idx,
    ):
        self.context = context
        self.target = target
        self.word_2_idx = word_2_idx

        self.idx_2_word = {v: k for k, v in word_2_idx.items()}
        self.vocabulary = list(word_2_idx.keys())

    @classmethod
    def from_raw(cls):
        sentences = cls._process_sentences(cls._load_sentences())

        word_2_idx = {}
        most_common_words = {
            x[0]: i
            for i, x in enumerate(
                Counter([word for sentence in sentences for word in sentence]).most_common()[:cls.vocab_size - 1]
            )
        }
        word_2_idx.update(most_common_words)
        word_2_idx[CONFIG["data"]["unk_token"]] = cls.vocab_size - 1

        context, target = cls._create_pairs(sentences, word_2_idx)

        return cls(context, target, word_2_idx)

    @classmethod
    def from_files(
        cls,
        tokenizer_path=CONFIG["data"]["tokenizer_path"],
        context_path=CONFIG["data"]["context_path"],
        target_path=CONFIG["data"]["target_path"],
        filter_unk_target=True
    ):
        with open(tokenizer_path) as f:
            word_2_idx = json.load(f)

        context = torch.LongTensor(torch.load(context_path))
        target = torch.LongTensor(torch.load(target_path))

        if filter_unk_target:
            target_not_unk_mask = torch.sum(target != len(word_2_idx) - 1, dim=1) == target.shape[1]
            target = target[target_not_unk_mask]
            context = context[target_not_unk_mask]

            context_not_unk_mask = torch.sum(context != len(word_2_idx) - 1, dim=1) == context.shape[1]
            target = target[context_not_unk_mask]
            context = context[context_not_unk_mask]

        return cls(context, target, word_2_idx)

    def save_to_files(
        self,
        tokenizer_path=CONFIG["data"]["tokenizer_path"],
        context_path=CONFIG["data"]["context_path"],
        target_path=CONFIG["data"]["target_path"]
    ):
        with open(tokenizer_path, 'w') as f:
            json.dump(self.word_2_idx, f)

        torch.save(self.context, context_path)
        torch.save(self.target, target_path)

    @staticmethod
    def _load_sentences():
        with open(CONFIG["data"]["uc_path"], "r") as f:
            file_text = f.read()
            sentences = file_text.split('\n')
            num_sentences = int(len(sentences) * CONFIG["data"]["uc_ratio"])
            sentences = sentences[:num_sentences]

        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) > 5:
                filtered_sentences.append(sentence)

        return filtered_sentences

    @staticmethod
    def _load_stopwords():
        all_stop_words = pd.read_csv(CONFIG["data"]["stopwords"], header=None, names=['stopwords'])
        all_stop_words = list(all_stop_words.stopwords)

        stop_words = []
        stop_phrases = []
        for stopword in all_stop_words:
            if " " not in stopword:
                stop_words.append(stopword)
            else:
                stop_phrases.append(stopword)

        return stop_words, stop_phrases

    @classmethod
    def _process_sentences(cls, sentences):
        processed_sentences = []
        stop_words, stop_phrases = cls._load_stopwords()

        for sentence in tqdm(sentences):
            sentence = sentence.lower()
            sentence = re.sub(r'[^а-щьюяіїґє\s]', '', sentence)

            for stop_phrase in stop_phrases:
                sentence = sentence.replace(stop_phrase, "")

            sentence = sentence.strip()

            if len(sentence) > 5:
                split_sentence = sentence.split()
                split_sentence = [word for word in split_sentence if word not in stop_words and len(word) > 3]

                if len(split_sentence) > CONFIG["data"]["window_size"] + 1:
                    processed_sentences.append(split_sentence)

        return processed_sentences

    @classmethod
    def _create_pairs(cls, sentences, word_2_idx):
        context = []
        target = []

        for sentence in sentences:
            encoded_sentence = np.array([word_2_idx.get(word, CONFIG["data"]["vocab_size"] - 1) for word in sentence])

            for i in range(len(encoded_sentence) - 1 - CONFIG["data"]["window_size"]):
                sentence_context = encoded_sentence[i:i + CONFIG["data"]["window_size"]]
                sentence_target = encoded_sentence[i + 1:i + 1 + CONFIG["data"]["window_size"]]

                context.append(sentence_context)
                target.append(sentence_target)

        context = torch.LongTensor(np.array(context))
        target = torch.LongTensor(np.array(target))

        return context, target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]
