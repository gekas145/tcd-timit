import torch
import sys
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from copy import deepcopy


class StackedBPLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([BPLSTM(input_size if i == 0 else 2*hidden_size, 
                                            hidden_size, 
                                            dropout if i < num_layers - 1 else 0.0) for i in range(num_layers)])

    def forward(self, inputs, input_length=None):
        outputs = self.layers[0](inputs, input_length)

        for i in range(1, len(self.layers)):
            outputs = self.layers[i](outputs, input_length)

        return outputs
    

class BPLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None

        self.weight_ih_forward = self.__uniform(input_size, 4*hidden_size)
        self.weight_hh_forward = self.__uniform(hidden_size, 4*hidden_size)
        self.weight_ch_forward = self.__uniform(3*hidden_size)
        self.bias_forward = self.__uniform(4*hidden_size)

        self.weight_ih_backward = self.__uniform(input_size, 4*hidden_size)
        self.weight_hh_backward = self.__uniform(hidden_size, 4*hidden_size)
        self.weight_ch_backward = self.__uniform(3*hidden_size)
        self.bias_backward = self.__uniform(4*hidden_size)

    def forward(self, inputs, input_length=None):
        max_length = inputs.shape[-2]
        if input_length is not None:
            lengths = input_length.view(len(input_length), 1)

        dims = list(inputs.shape)
        dims.pop(-2)
        dims[-1] = self.hidden_size

        h_forward, c_forward = torch.zeros(dims), torch.zeros(dims)
        h_backward, c_backward = torch.zeros(dims), torch.zeros(dims)

        forwards = []
        backwards = []

        for i in range(max_length):
            h_forward, c_forward = self.__step(inputs[..., i, :], h_forward, c_forward, 
                                               self.weight_ih_forward, 
                                               self.weight_hh_forward, 
                                               self.weight_ch_forward,
                                               self.bias_forward)
            
            h_backward, c_backward = self.__step(inputs[..., -i-1, :], h_backward, c_backward, 
                                                 self.weight_ih_backward, 
                                                 self.weight_hh_backward, 
                                                 self.weight_ch_backward,
                                                 self.bias_backward)
            if input_length is not None:
                mask = lengths >= max_length - i - 1
                h_backward *= mask
            
            forwards.append(h_forward)
            backwards.append(h_backward)

        forwards = torch.stack(forwards)
        backwards = torch.flip(torch.stack(backwards), dims=(0,))
        outputs = torch.cat((forwards, backwards), dim=-1)
        if len(outputs.shape) == 3:
            outputs = torch.permute(outputs, (1, 0, 2))

        if self.dropout is not None:
            self.dropout(outputs)
        
        return outputs

    def __uniform(self, *size):
        return nn.parameter.Parameter((2 * torch.rand(size) - 1)/np.sqrt(self.hidden_size))
    
    def __step(self, x, h, c, weight_ih, weight_hh, weight_ch, bias):

        f = torch.sigmoid(torch.matmul(x, weight_ih[:, :self.hidden_size]) + \
                          c * weight_ch[:self.hidden_size] + \
                          torch.matmul(h, weight_hh[:, :self.hidden_size]) + bias[:self.hidden_size])

        i = torch.sigmoid(torch.matmul(x, weight_ih[:, self.hidden_size:2*self.hidden_size]) + \
                          c * weight_ch[self.hidden_size:2*self.hidden_size] + \
                          torch.matmul(h, weight_hh[:, self.hidden_size:2*self.hidden_size]) + bias[self.hidden_size:2*self.hidden_size])
        
        g = torch.tanh(torch.matmul(x, weight_ih[:, 2*self.hidden_size:3*self.hidden_size]) + \
                       torch.matmul(h, weight_hh[:, 2*self.hidden_size:3*self.hidden_size]) + bias[2*self.hidden_size:3*self.hidden_size])
        
        c_new = f * c + i * g
        
        o = torch.sigmoid(torch.matmul(x, weight_ih[:, 3*self.hidden_size:4*self.hidden_size]) + \
                          c_new * weight_ch[2*self.hidden_size:3*self.hidden_size] + \
                          torch.matmul(h, weight_hh[:, 3*self.hidden_size:4*self.hidden_size]) + bias[3*self.hidden_size:4*self.hidden_size])

        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class TokenSet:

    def __init__(self, words):
        self.vocab = [word for word in words]

    def __getitem__(self, key):
        return self.vocab[key]
    
    def __iter__(self):
        for i, word in enumerate(self.vocab):
            yield i, word

    def __len__(self):
        return len(self.vocab)


class Normalizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=False)
        self.std = nn.parameter.Parameter(torch.tensor(1.0), requires_grad=False)

    def adapt(self, data):
        self.mean.copy_(torch.mean(data))
        self.std.copy_(torch.std(data))

    def forward(self, data):
        return (data - self.mean)/self.std


class TcdTimitDataset(Dataset):

    def __init__(self, waves, labels, vocab):
        self.waves = waves
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.waves[idx], self.labels[idx]


def collate_fn(data):

    # stack waves
    waves = torch.stack([d[0] for d in data])

    # get waves lengths
    waves_lengths = tuple(d[0].shape[0] for d in data)

    # stack labels
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(d[1], dtype=torch.uint8) for d in data], batch_first=True)

    # get labels lengths
    labels_lengths = tuple(len(d[1]) for d in data)

    return waves, waves_lengths, labels, labels_lengths


class Prefix:

    def __init__(self, prefix, score, blank_end):
        self.prefix = prefix

        self.blank_score = score if blank_end else 0.0
        self.not_blank_score = 0.0 if blank_end else score

        self.blank = blank_end
        self.not_blank = not blank_end

    def __hash__(self):
        return hash(self.prefix)
    
    def __eq__(self, other_prefix):
        return self.prefix == other_prefix.prefix
    
    @staticmethod
    def __add_log_prob(p1, p2):
        return max(p1, p2) + np.log(1 + np.exp(-np.abs(p1 - p2)))

    @property
    def score(self):
        if self.blank and self.not_blank:
            return Prefix.__add_log_prob(self.blank_score, self.not_blank_score)
        
        if self.blank:
            return self.blank_score

        return self.not_blank_score

    def add(self, char, char_score, is_blank):
        if is_blank:
            return (Prefix(self.prefix, self.score + char_score, True), )

        if self.blank and self.not_blank and self.prefix[-1] == char:
            return (Prefix(self.prefix + char, self.blank_score + char_score, False),
                    Prefix(self.prefix, self.not_blank_score + char_score, False))
        
        if self.not_blank:
            new_prefix = self.prefix if self.prefix[-1] == char else self.prefix + char
            return (Prefix(new_prefix, self.score + char_score, False), )
        
        
        return (Prefix(self.prefix + char, self.score + char_score, False), )
        

    def merge(self, other_prefix):
        if self.blank and other_prefix.blank:
            self.blank_score = Prefix.__add_log_prob(self.blank_score, other_prefix.blank_score)
        elif other_prefix.blank:
            self.blank = True
            self.blank_score = other_prefix.blank_score
        
        if self.not_blank and other_prefix.not_blank:
            self.not_blank_score = Prefix.__add_log_prob(self.not_blank_score, other_prefix.not_blank_score)
        elif other_prefix.not_blank:
            self.not_blank = True
            self.not_blank_score = other_prefix.not_blank_score


def beam_search(char_scores, vocab, n):
    ''' 
    Performs beam search for CTC loss inference.

    Args:

    char_scores   - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,

    vocab         - string containing each character from vocabulary, vocab[0] has to be the blank char,

    n             - width of beam search.

    Return:

    str.

    '''

    blank_char = vocab[0]

    prefixes = [Prefix(vocab[i] if i > 0 else "", 
                       char_scores[0][i], 
                       i == 0) for i in range(len(vocab))]

    for i in range(1, len(char_scores)):

        new_prefixes = {}

        for char_id, char in enumerate(vocab):
            for prefix in prefixes:
                possibilities = prefix.add(char, char_scores[i][char_id], char == blank_char)

                for p in possibilities:
                    if p in new_prefixes:
                        new_prefixes[p].merge(p)
                    else:
                        new_prefixes[p] = p

        new_prefixes = [new_prefixes[prefix] for prefix in new_prefixes]
        new_prefixes.sort(key=lambda x: x.score, reverse=True)
        prefixes = new_prefixes[:n]

    prefixes.sort(key=lambda x: x.score, reverse=True)

    return prefixes[0].prefix



def decode_sentence(char_scores, vocab, decode_fun, silence_threshold=0.998, **kwargs):
    ''' 
    Decodes CTC loss by splitting main sequence by silence steps and using chosen decoding function on those smaller subsequences.

    Args:

    char_scores        - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,

    vocab              - string containing each character from vocabulary, vocab[0] has to be the blank char,

    decode_fun         - decoding function to use on subsequences between silence steps, e.g. beam_search; expected to take char_scores like object, vocab and return str,

    silence_threshold  - minimal prob of silence char required for step to be considered silent,

    kwargs             - keyword arguments for decode_fun.

    Return:

    str.

    '''

    sentence = ""

    silence = np.argwhere(np.exp(char_scores[:, 0]) >= silence_threshold)
    silence = np.squeeze(silence)
    if silence[-1] != char_scores.shape[0] - 1:
        silence = np.append(silence, char_scores.shape[0])

    start, end = -1, -1

    for idx in silence:
        end = idx

        if end - start == 1:
            start += 1
            continue
        
        word = decode_fun(char_scores[start+1:end, :], vocab, **kwargs)
        sentence += word if len(sentence) == 0 else " " + word

        start = end

    return sentence


def decode_token_passing(char_scores, vocab, token_set):
    ''' 
    Performs token passing for CTC loss inference.

    Args:

    char_scores   - 2d array of shape (m, len(vocab)), where m is number of timesteps, containing chars log probs,

    vocab         - string containing each character from vocabulary, vocab[0] has to be the blank char,

    token_set     - list of all possible words.

    Return:

    1d array with integers being indexes of words from token_set.

    '''

    tok = dict()

    for i, word in token_set:
        tok[(i, 0, 0)] = [char_scores[0][0], [i]]
        tok[(i, 1, 0)] = [char_scores[0][vocab.index(word[0])], [i]]

        if len(word) == 1:
            tok[(i, -1, 0)] = [*tok[(i, 1, 0)], False]
        else:
            tok[(i, -1, 0)] = [-np.inf, [], False]

        for s in range(2, 2*len(word) + 1):
            tok[(i, s, 0)] = [-np.inf, []]

    for t in range(1, len(char_scores)):

        best_finished_token = max(range(len(token_set)), key=lambda w: tok[(w, -1, t-1)][0])

        for i, word in token_set:

            if tok[(best_finished_token, -1, t-1)][0] == -np.inf:
                best, blank = None, False
            else:
                score, path, blank = tok[(best_finished_token, -1, t-1)]
                best = [score, path + [i]]

            for s in range(2*len(word) + 1):

                P = [tok[(i, s, t-1)]]

                if s == 0 and best is not None:
                    P.append(best)

                if s == 1 and best is not None and (blank or token_set[best_finished_token][-1] != word[0]):
                    P.append(best)

                if s > 0:
                    P.append(tok[(i, s-1, t-1)])

                if s > 1 and s%2 == 1 and word[s//2 - 1] != word[s//2]:
                    P.append(tok[(i, s-2, t-1)])

                max_token = deepcopy(max(P, key=lambda token: token[0]))
                idx = 0 if s%2 == 0 else vocab.index(word[s//2])
                max_token[0] += char_scores[t][idx]
                tok[(i, s, t)] = max_token

            if tok[(i, 2*len(word) - 1, t)] > tok[(i, 2*len(word), t)]:
                tok[(i, -1, t)] = (*tok[(i, 2*len(word) - 1, t)], False)
            else:
                tok[(i, -1, t)] = (*tok[(i, 2*len(word), t)], True)

    best_w = max(range(len(token_set)), key=lambda w: tok[(w, -1, len(char_scores) - 1)][0])

    return tok[(best_w, -1, len(char_scores) - 1)][1]


# Print iterations progress
# this code was taken from
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?page=1&tab=votes#tab-top
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    # Print New Line on Complete
    if iteration == total:
        print()









if __name__ == "__main__":

    token_set = TokenSet(["abb", "aab", "abc"])

    print(decode_token_passing([[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], 
                                "*abc", 
                                token_set))


