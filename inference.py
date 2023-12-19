import pickle
import torch
import torchaudio
import zipfile
import re
import config as c
import matplotlib.pyplot as plt
from model import Model
from dataset import training_dataset, training_files, validation_dataset, validation_files, testing_dataset, testing_files
from utils import beam_search, printProgressBar

files = testing_files
dataset = testing_dataset
dataset_name = "validation_phonemes"

vocab = training_dataset.dataset.vocab.copy()
print("Vocab:", training_dataset.dataset.vocab)
vocab = [" " + vocab[i] for i in range(len(vocab))]

model = Model(c.input_dim, c.lstm_hidden_dim, c.num_lstm_layers, c.lstm_dropout, c.dropout1, c.decoder_dim, c.dropout2, len(training_dataset.dataset.vocab))

with open("tcd-timit/checkpoints/12 checkpoint tr 22.54 val 31.73.pt", "rb") as f:
    dict = pickle.load(f)

model.load_state_dict(dict)
model.eval()

timit = zipfile.ZipFile("tcd-timit/tcd-timit.zip", "r")

per = 0.0
length = 0

printProgressBar(0, len(files))
with torch.inference_mode():

    for n in range(len(files)):

        log_probs = model(dataset.dataset.waves[n]).numpy()
        decoded_sentence = beam_search(log_probs, vocab, 5)
        decoded_sentence = re.sub("^\s", "", decoded_sentence)
        decoded_sentence = decoded_sentence.split(" ")
        
        with timit.open(f"phonemes/{files[n]}.txt") as f:
            text = f.read().decode().split(" ")

        # with open(f"tcd-timit/predictions/{dataset_name}/{files[n]}.txt", "w") as f:
            # f.write(f"Original: {text}\nPredicted: {decoded_sentence}")

        per += torchaudio.functional.edit_distance(decoded_sentence, text)
        length += len(text)

        printProgressBar(n + 1, len(files))

timit.close()

print(f"PER: {per/length*100:.2f}%")

