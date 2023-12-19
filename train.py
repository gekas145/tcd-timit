import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import config as c
from dataset import training_dataset, validation_dataset
from model import Model

def get_loss(waves, waves_lengths, labels, labels_lengths):

    pred = model(waves)
    pred = torch.permute(pred, (1, 0, 2))

    return criterion(pred, labels, waves_lengths, labels_lengths)

def run_inference():
    model.eval()
    with torch.inference_mode():
        training_loss, validation_loss = 0.0, 0.0

        for training_dt, validation_dt in zip(training_dataset, validation_dataset):
            training_loss += get_loss(*training_dt)
            validation_loss += get_loss(*validation_dt)

        training_loss = training_loss.data/len(validation_dataset.dataset)
        validation_loss = validation_loss.data/len(validation_dataset.dataset)

    model.train()

    return training_loss, validation_loss

model = Model(c.input_dim, c.lstm_hidden_dim, c.num_lstm_layers, c.lstm_dropout, c.dropout1, c.decoder_dim, c.dropout2, len(training_dataset.dataset.vocab))
for param in model.parameters():
    if param.requires_grad:
        param.register_hook(lambda grad: torch.clamp(grad, min=-c.clip_value, max=c.clip_value))


optimizer = optim.Adam(model.parameters(), lr=c.learning_rate)
criterion = nn.CTCLoss(reduction="sum")

model.normalizer.adapt(training_dataset.dataset.waves)


for epoch in range(1, c.nepochs+1):

    for waves, waves_lengths, labels, labels_lengths in training_dataset:

        optimizer.zero_grad()

        loss = get_loss(waves, waves_lengths, labels, labels_lengths)
        loss /= waves.shape[0]
        loss.backward()

        optimizer.step()
    
    if epoch % c.verbosity == 0:

        training_loss, validation_loss = run_inference()

        print(f"[Epoch: {epoch}] train loss: {training_loss:.2f}, validation loss: {validation_loss:.2f}")

        with open(f"tcd-timit/checkpoints/{epoch//c.verbosity} checkpoint tr {training_loss:.2f} val {validation_loss:.2f}.pt", "wb") as f:
            pickle.dump(model.state_dict(), f)


if c.nepochs % c.verbosity != 0:
    training_loss, validation_loss = run_inference()
    with open(f"tcd-timit/checkpoints/final checkpoint tr {training_loss:.2f} val {validation_loss:.2f}.pt", "wb") as f:
        pickle.dump(model.state_dict(), f)



