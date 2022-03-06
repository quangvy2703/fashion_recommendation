
from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransactionsEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_directions, num_layers=1):
        super(TransactionsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.n_directions = n_directions

        self.fc = nn.Linear(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, int(self.hidden_size / self.n_directions),  self.num_layers, batch_first=True, bidirectional=True if self.n_directions == 2 else False)
        # self.lstm = nn.LSTM(
        #     self.hidden_size,
        #     int(self.hidden_size/2),  # Bi-directional processing will ouput vectors of double size, therefore I reduced output dimensionality
        #     num_layers=self.num_layers,
        #     batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
        #     bidirectional=True
        # )

    def forward(self, input, hidden):
        # input: (batch_size, seq_length, article_features_length)
        output = self.fc(input)
        # output: (batch_size, seq_length, hidden_size)
        # output = output.unsqueeze(0)

        # print(output.size(), hidden.size())
        output, hidden = self.gru(output, hidden)
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (h: (num_layers*directions, batch_size, hidden_size),
        #          c: (num_layers*directions, batch_size, hidden_size))
        # output, hidden = self.lstm()
        return output, hidden

    def init_hidden(self, batch_size):
        hidden =  torch.zeros(2, self.num_layers * self.n_directions, batch_size, int(self.hidden_size / self.n_directions), device=device)
        hidden = device.to(device)
        return hidden


class CustomerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomerEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.fc(input)
        # output = output * transaction_encoder_output
        output = self.out(output)
        # output = self.softmax(output)
        return output
    


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_SEQUENCE_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, customer_encoder_output, transaction_encoder_outputs, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)

        # print(customer_encoder_output.size(),  transaction_encoder_outputs.size(), hidden.size())
        attn_weights = F.softmax(
            self.attn(torch.cat((customer_encoder_output, hidden[0]), 1)), dim=1)
        #att_weights [batch_size x len_seq]
        #encoder_output [batch_size x len_seq x hidden]
        # print(attn_weights.size(), attn_weights.unsqueeze(1).size(), transaction_encoder_outputs.size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                transaction_encoder_outputs)

        # print(customer_encoder_output.size(), attn_applied.size())
        output = torch.cat((customer_encoder_output, attn_applied.squeeze(1)), 1)
        # print(output.size())
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # output, hidden = self.gru(output, hidden)
        output = self.out(output)
        # print(output)
        # output = F.log_softmax(output, dim=1)
        # print(output)
        return output[0], attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, 1, self.hidden_size, device=device)