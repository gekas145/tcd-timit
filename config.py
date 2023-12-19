


# data params
timesteps = 300
n_fft = 1024
win_length = 1024
hop_length = 512
n_mels = 32
n_mfcc = 16
wav_rate = 48000

# model params
input_dim = 15
num_lstm_layers = 3
lstm_hidden_dim = 256
decoder_dim = 512
lstm_dropout = 0.6
dropout1 = 0.6
dropout2 = 0.6

# training params
batch_size = 16
nepochs = 100
verbosity = 2
learning_rate = 0.003
clip_value = 20

