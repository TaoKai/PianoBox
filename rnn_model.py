import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PianoResLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(PianoResLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm0 = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.activ = nn.ReLU()
        self.norm = nn.LayerNorm([12, self.hidden_size])
        self.h_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x, h_init):
        h0 = h_init
        res0 = x
        x1, h1 = self.lstm0(x, h0)
        x1 = self.activ(x1)
        x2, h2 = self.lstm1(x1, h1)
        x2 = x2+res0
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = self.activ(x2)
        x2 = x2.permute(1, 0, 2)
        h2 = (self.h_norm(h2[0]), self.h_norm(h2[1]))
        return x2, h2


class PianoBox(nn.Module):
    def __init__(self, embed_size, note_num, off_num):
        super(PianoBox, self).__init__()
        self.note_num = note_num
        self.off_num = off_num
        self.hidden_size = embed_size*2
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(note_num, embed_size)
        self.off_embeddings = nn.Embedding(off_num, embed_size)
        self.prLstm0 = PianoResLSTM(self.hidden_size)
        self.prLstm1 = PianoResLSTM(self.hidden_size)
        self.prLstm2 = PianoResLSTM(self.hidden_size)
        self.prLstm3 = PianoResLSTM(self.hidden_size)
        self.prLstm4 = PianoResLSTM(self.hidden_size)
        self.prLstm5 = PianoResLSTM(self.hidden_size)
        self.prLstm6 = PianoResLSTM(self.hidden_size)
        self.prLstm7 = PianoResLSTM(self.hidden_size)
        self.prLstm8 = PianoResLSTM(self.hidden_size)
        self.prLstm9 = PianoResLSTM(self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, 1)
        self.pitch_cls = nn.Linear(self.hidden_size, note_num)
        self.pitch_cls2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.olv_reg = nn.Linear(self.hidden_size, off_num)
        self.olv_reg2 = nn.Linear(self.hidden_size, self.hidden_size)
    
    def initHidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_size, device=device), 
        torch.randn(1, batch_size, self.hidden_size, device=device))

    def forward(self, pitches, olv_feats, h_init=None):
        pitches = pitches.long()
        olv_feats = olv_feats.long()
        h_init = self.initHidden(pitches.shape[0]) if h_init is None else h_init
        embs = self.embeddings(pitches)
        off_embs = self.off_embeddings(olv_feats)
        x_input = torch.cat([embs, off_embs], dim=2).permute(1, 0, 2)
        x0, h0 = self.prLstm0(x_input, h_init)
        x1, h1 = self.prLstm1(x0, h0)
        x2, h2 = self.prLstm2(x1, h1)
        x3, h3 = self.prLstm3(x2, h2)
        x4, h4 = self.prLstm4(x3, h3)
        x5, h5 = self.prLstm5(x4, h4)
        x6, h6 = self.prLstm6(x5, h5)
        x7, h7 = self.prLstm7(x6, h6)
        x8, h8 = self.prLstm8(x7, h7)
        x9, h9 = self.prLstm9(x8, h8)
        x9 = x9.permute(1, 0, 2)
        attn_vec = F.softmax(self.attn(x9).reshape(-1, 12), dim=1).reshape([-1, 12, 1])
        x9 = F.relu((attn_vec*x9).sum(axis=1))
        pitch_prob = F.softmax(self.pitch_cls(self.pitch_cls2(x9)), dim=1)
        olv_vec = F.softmax(self.olv_reg(self.olv_reg2(x9)), dim=1)
        return pitch_prob, olv_vec, h9
        # return pitch_prob


if __name__ == "__main__":
    piano = PianoBox(512, 88, 1000)
    olv_feats = torch.randint(0, 1000, [16, 12])
    pitches = torch.randint(0, 88, [16, 12])
    x = piano(pitches, olv_feats)
    print(x, x.shape)