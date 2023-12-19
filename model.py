import torch.nn as nn
from utils import Normalizer, StackedBPLSTM


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, lstm_dropout, dropout1, decoder_dim, dropout2, output_dim):
        super().__init__()

        self.normalizer = Normalizer()
        
        # self.lstm = StackedBPLSTM(input_size=input_dim,
        #                           hidden_size=hidden_dim,
        #                           dropout=lstm_dropout,
        #                           num_layers=num_layers)
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=lstm_dropout,
                            batch_first=True,
                            bidirectional=True)
        
        self.dropout1 = nn.Dropout(dropout1)
        
        self.linear1 = nn.Linear(2*hidden_dim, decoder_dim)

        self.dropout2 = nn.Dropout(dropout2)

        self.linear2 = nn.Linear(decoder_dim, output_dim)

    def forward(self, inputs):
        outputs = self.normalizer(inputs)
        outputs, _ = self.lstm(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.linear2(outputs)
        return nn.functional.log_softmax(outputs, dim=-1)



