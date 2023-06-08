from training.train import TrainingLoop
from model.encoder_decoder import EncoderDecoder
from dataset.bruk import BrukDataset


# bruk = BrukDataset.from_files()
# print(bruk.word_2_idx)
bruk = BrukDataset.from_raw()
bruk.save_to_files()

model = TrainingLoop(EncoderDecoder()).run(bruk)
