import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PianoCell(nn.Module):
    def __init__(self, note_size, off_size, note_num):
        super(PianoCell, self).__init__()
        self.note_size = note_size
        self.off_size = off_size
        self.hidden_size = note_size+off_size
        self.note_num = note_num
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.note_linear = nn.Linear(self.hidden_size, note_size)
        self.note_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.offset_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.offset_reg = nn.Linear(self.hidden_size, 1)
        self.embeddings = nn.Embedding(note_num, self.note_size)
        self.offset_vec = nn.Linear(1, self.off_size)
        self.smooth = nn.SmoothL1Loss()
        self.embedloss = MultiEmbeddingLoss()
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros([batch_size, self.hidden_size])
        c0 = torch.zeros([batch_size, self.hidden_size])
        return (h0, c0)

    def forward(self, note_inputs, off_inputs, hc=None):
        batch_size = note_inputs.shape[0]
        if hc is None:
            hc = self.init_hidden(batch_size)
        off_inputs = off_inputs.reshape(-1, 1)
        off_inputs = self.offset_vec(off_inputs)
        embs = self.embeddings(note_inputs)
        x = torch.cat([embs, off_inputs], dim=-1)
        ho, co = self.lstm(x, hc)
        note_emb = self.note_linear(self.note_hidden(ho))
        off_reg = F.relu(self.offset_reg(self.offset_hidden(ho)))
        off_reg = off_reg.reshape([-1])
        return note_emb, off_reg, (ho, co)
    
    def loss(self, note_emb, off_reg, labels, sample_neg_labels, off_labels):
        note_loss = self.embedloss(note_emb, labels, sample_neg_labels, self.embeddings)
        off_loss = self.smooth(off_reg, off_labels)
        total_loss = note_loss+off_loss
        print("total_loss:", total_loss, 'note_loss:', note_loss, 'off_loss:', off_loss)
        return total_loss

class MultiEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(MultiEmbeddingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred_embs, labels, sample_neg_labels, embeddings):
        pos_embs = embeddings(labels.long())
        neg_embs = embeddings(sample_neg_labels.long())
        batch_size = pred_embs.shape[0]
        neg_pred_embs = pred_embs.reshape(batch_size, 1, -1)
        neg_cos_sim_top = (neg_pred_embs*neg_embs).sum(axis=-1)
        neg_cos_sim_bottom = torch.sqrt(torch.pow(neg_pred_embs, 2).sum(axis=-1))*torch.sqrt(torch.pow(neg_embs, 2).sum(axis=-1))
        neg_cos_sim = neg_cos_sim_top/neg_cos_sim_bottom
        neg_cos_loss = F.relu(neg_cos_sim-self.margin).mean()
        pos_cos_sim_top = (pos_embs*pred_embs).sum(axis=-1)
        pos_cos_sim_bottom = torch.sqrt(torch.pow(pos_embs, 2)).sum(axis=-1)*torch.sqrt(torch.pow(pred_embs, 2).sum(axis=-1))
        pos_cos_loss = 1-pos_cos_sim_top/pos_cos_sim_bottom
        pos_cos_loss = pos_cos_loss.mean()
        total_loss = pos_cos_loss+neg_cos_loss
        return total_loss

def get_random_neg_labels(labels, emb_len, sample_num):
    emb_all = list(np.arange(emb_len))
    for l in labels:
        if l in emb_all:
            emb_all.remove(l)
    random.shuffle(emb_all)
    return emb_all[:sample_num]

if __name__ == "__main__":
    from note_process2 import Note
    note_train = Note('raw_pieces.json')
    pianoCell = PianoCell(768, 256, note_train.embedding_len)
    note_inputs, off_inputs, note_labels, off_labels, mask = note_train.next()
    neg_labels = get_random_neg_labels(note_labels, note_train.embedding_len, 5000)
    neg_labels = torch.tensor(neg_labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32).reshape(-1, 1)
    hc = pianoCell.init_hidden(note_train.batch_size)
    hc = (hc[0]*mask, hc[1]*mask)
    note_inputs = torch.tensor(note_inputs, dtype=torch.long)
    off_inputs = torch.tensor(off_inputs, dtype=torch.float32)
    note_labels = torch.tensor(note_labels, dtype=torch.long)
    off_labels = torch.tensor(off_labels, dtype=torch.float32)
    note_emb, off_reg, hc = pianoCell(note_inputs, off_inputs)
    cost = pianoCell.loss(note_emb, off_reg, note_labels, neg_labels, off_labels)
    print(cost)