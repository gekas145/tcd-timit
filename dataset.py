import torch
import torchvision
import torchaudio
import zipfile
import re
import numpy as np
import config as c
import matplotlib.pyplot as plt
from utils import TcdTimitDataset, collate_fn, Normalizer
from functools import reduce

def load_data(files):

    waves = []
    sentences = []

    for file in files:
        with timit.open("phonemes/" + file + ".txt") as f:
            sentences.append(f.read().decode().split(" "))

        with timit.open("waves/" + file + ".wav") as f:
            wave, _ = torchaudio.load(f)
            wave = mfcc(wave)[:, 1:, :] # omit 1st mfcc coeff
            wave = resize(wave)
            wave = torch.squeeze(wave, 0)
            wave = torch.permute(wave, (1, 0))
            waves.append(wave)

    waves = torch.stack(waves)

    vocab = reduce(lambda x, y: x + y, sentences)
    vocab = np.unique(vocab).tolist()
    vocab.sort()
    vocab.insert(0, "*")

    sentences = [[vocab.index(phoneme) for phoneme in sentence] for sentence in sentences]

    return torch.utils.data.DataLoader(TcdTimitDataset(waves, sentences, vocab),
                                       batch_size=c.batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn)

def load_filenames(file):
    with open(file, "r") as f:
        filenames = f.readlines()

    return [re.sub("\n", "", fname) for fname in filenames]


timit = zipfile.ZipFile("tcd-timit/tcd-timit.zip", "r")

mfcc = torchaudio.transforms.MFCC(sample_rate=c.wav_rate, 
                                  n_mfcc=c.n_mfcc, 
                                  melkwargs={"n_fft": c.n_fft, 
                                             "win_length": c.win_length, 
                                             "hop_length": c.hop_length, 
                                             "n_mels": c.n_mels})

resize = torchvision.transforms.Resize((c.input_dim, c.timesteps), antialias=True)

training_files = load_filenames("tcd-timit/data_division/training.txt")
training_dataset = load_data(training_files)

validation_files = load_filenames("tcd-timit/data_division/validation.txt")
validation_dataset = load_data(validation_files)

testing_files = load_filenames("tcd-timit/data_division/testing.txt")
testing_dataset = load_data(testing_files)

timit.close()


if __name__ == "__main__":

    normalizer = Normalizer()
    normalizer.adapt(validation_dataset.dataset.waves)
    print(normalizer.mean, normalizer.std)

    plt.figure(figsize = (20, 10))
    plt.imshow(normalizer(validation_dataset.dataset.waves[0]))
    plt.show()

    print("Waves shape:", validation_dataset.dataset.waves.shape)
    print("Vocab:", validation_dataset.dataset.vocab)
    print("Vocab size:", len(validation_dataset.dataset.vocab))

    for waves, waves_lengths, labels, labels_lengths in validation_dataset:
        print("Batch waves shape:", waves.shape)
        print("Batch waves lengths:", waves_lengths)
        print("Batch labels shape:", labels.shape)
        print("Batch labels lengths:", labels_lengths)
        break







