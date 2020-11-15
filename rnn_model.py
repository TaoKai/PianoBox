import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PianoBox(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PianoBox, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru0 = nn.GRU(hidden_size, hidden_size, 1)
        self.gru1 = nn.GRU(hidden_size, hidden_size, 1)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1)
        self.gru3 = nn.GRU(hidden_size, hidden_size, 1)
        self.linear = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax()
        self.attn = nn.Linear(hidden_size, 1)
        self.group_cls = nn.Linear(hidden_size, 9)
        self.pitch_cls = nn.Linear(hidden_size, 12)
        self.chord_cls = nn.Linear(hidden_size, 2)
        self.olv_reg = nn.Linear(hidden_size, 3)
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
    def forward(self, x, h_init=None):
        batch_size = x.shape[0]
        h0 = self.initHidden(batch_size) if h_init is None else h_init
        x = self.linear(x)
        res0 = x.reshape([12, -1, self.hidden_size])
        x1, h1 = self.gru0(res0, h0)
        x1 = x1.reshape([-1, self.hidden_size, 12])
        x1 = self.bn(x1).reshape([12, -1, self.hidden_size])
        h1 = h1.reshape([-1, self.hidden_size, 1])
        h1 = self.bn(h1).reshape([1, -1, self.hidden_size])
        x2, h2 = self.gru1(x1, h1)
        x2 += res0
        x2 = F.relu(x2)
        x2 = x2.reshape([-1, self.hidden_size, 12])
        x2 = self.bn(x2).reshape([12, -1, self.hidden_size])
        h2 = h2.reshape([-1, self.hidden_size, 1])
        h2 = self.bn(h2).reshape([1, -1, self.hidden_size])
        x2 = self.drop(x2)
        res1 = x2
        x3, h3 = self.gru2(res1, h2)
        x3 = x3.reshape([-1, self.hidden_size, 12])
        x3 = self.bn(x3).reshape([12, -1, self.hidden_size])
        h3 = h3.reshape([-1, self.hidden_size, 1])
        h3 = self.bn(h3).reshape([1, -1, self.hidden_size])
        x4, h4 = self.gru3(x3, h3)
        x4 += res1
        x4 = x4.reshape([-1, self.hidden_size, 12])
        x4 = self.bn(x4).reshape([12, -1, self.hidden_size])
        h4 = h4.reshape([-1, self.hidden_size, 1])
        h4 = self.bn(h4).reshape([1, -1, self.hidden_size])
        x4 = self.drop(x4)
        x4 = x4.reshape([-1, 12, self.hidden_size])
        attn_vec = self.attn(x4)
        attn_vec = self.softmax(attn_vec.reshape([-1, 12])).reshape([-1, 12, 1])
        x4 = attn_vec*x4
        x_out = F.relu(x4.sum(axis=1))
        group_prob = self.softmax(self.group_cls(x_out))
        pitch_prob = self.softmax(self.pitch_cls(x_out))
        chord_prob = self.softmax(self.chord_cls(x_out))
        olv_vec = F.relu(self.olv_reg(x_out))
        return group_prob, pitch_prob, chord_prob, olv_vec


if __name__ == "__main__":
    piano = PianoBox(6, 128)
    x = torch.randn(16, 12, 6)
    x = piano(x)
    print(x, x.size())