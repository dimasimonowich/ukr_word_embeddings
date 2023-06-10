from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CONFIG
import os
import stanza
import re
from tqdm import tqdm
import json


class BrukDataset(Dataset):
    vocab_size = CONFIG["data"]["vocab_size"]
    left_window_size = CONFIG["cbow"]["left_window_size"]

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
            target_not_unk_mask = target != len(word_2_idx) - 1
            target = target[target_not_unk_mask]
            context = context[target_not_unk_mask]

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
        sentences = []

        filenames = os.listdir(CONFIG["data"]["bruk_path"])[:CONFIG["data"]["train_size"] + CONFIG["data"]["test_size"]]

        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(CONFIG["data"]["bruk_path"], filename)

                with open(file_path, "r") as file:
                    file_text = file.read()
                    sentences.extend(re.split(r'\.|!|\?|\n', file_text))

        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) > 0:
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
        nlp = stanza.Pipeline(lang='uk', processors='tokenize,lemma')

        for sentence in tqdm(sentences):
            sentence = sentence.lower()
            sentence = re.sub(r'[^а-щьюяіїґє\s]', '', sentence)

            for stop_phrase in stop_phrases:
                sentence = sentence.replace(stop_phrase, "")

            sentence = sentence.strip()

            if len(sentence) > 0:
                sentence = ' '.join([word.lemma.lower() for word in nlp(sentence).sentences[0].words])
                split_sentence = sentence.split()
                split_sentence = [word for word in split_sentence if word not in stop_words]

                if len(split_sentence) > 0:
                    processed_sentences.append(split_sentence)

        return processed_sentences

    @classmethod
    def _create_pairs(cls, sentences, word_2_idx):
        context = []
        target = []

        for sentence in sentences:
            for i in range(len(sentence) - cls.left_window_size):
                target_location = i + cls.left_window_size

                context_words = sentence[i: target_location]
                target_word = sentence[target_location]

                context_idxes = [word_2_idx.get(c, cls.vocab_size - 1) for c in context_words]
                target_idxes = word_2_idx.get(target_word, cls.vocab_size - 1)

                context.append(context_idxes)
                target.append(target_idxes)

        context = torch.LongTensor(context)
        target = torch.LongTensor(target)

        return context, target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]
