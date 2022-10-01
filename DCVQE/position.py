
import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        ##   d_model   dimension of features，  max_len : sequence max length
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LocalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, time_length):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, time_length).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / time_length)).exp()

        for index in range(max_len):
            pe[index]

if __name__ =='__main__':

    pe = PositionalEmbedding(10, 20)
    inp = torch.rand(2,20,10)
    print(pe(inp).shape)
    print(pe(inp))
