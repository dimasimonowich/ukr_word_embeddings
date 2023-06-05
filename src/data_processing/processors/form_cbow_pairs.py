from data_processing.data_pocessor import DataProcessor
import numpy as np
from config import CONFIG


class FormCBOWPairs(DataProcessor):
    def __init__(self):
        self.window = CONFIG["padding"]["window"]
        self.pad_value = CONFIG["padding"]["pad_value"]

    def _process(self, sentences, *args, **kwargs):
        pairs = []

        for sentence in sentences:
            idx_sentences.append(self.__form_pairs(sentence))

        return idx_sentences

    def __form_pairs(self, sentence):
        sentence_len = len(sentence)

        for i, word in enumerate(sentence):
            if sentence[i] != self.pad_value:
                target = sentence[i]

                for j in range(i - self.window, i + self.window + 1):
                    if j != i and sentence_len - 1 >= j >= 0 and sentence[j] != 0:
                        context = sentence[j]
                        training_data = np.append(training_data, [[target, context]], axis=0)

    @staticmethod
    def __get_indexing(sentences):
        idx_2_word = {}
        word_2_idx = {}
        temp = []

        i = 1
        for sentence in sentences:
            for word in sentence.split():
                if word not in temp:
                    temp.append(word)
                    idx_2_word[i] = word
                    word_2_idx[word] = i
                    i += 1

        return idx_2_word, word_2_idx
