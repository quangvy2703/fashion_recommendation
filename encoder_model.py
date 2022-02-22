
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransactionsEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransactionsEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        embedded = self.fc(input).view(1, 1, -1)
        output = embedded
        # print(output.size())
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
    


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initInput(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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

        # print(customer_encoder_output.size(),  hidden.size(), hidden[0].size())
        attn_weights = F.softmax(
            self.attn(torch.cat((customer_encoder_output, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 transaction_encoder_outputs.unsqueeze(0))

        print(transaction_encoder_outputs.size(), attn_applied.size())
        output = torch.cat((transaction_encoder_outputs[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)