import torch
from torch.optim import Adam
from single_rnn import PianoCell, get_random_neg_labels
from note_process2 import Note

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, step_num, data_path):
    note_train = Note(data_path)
    pianoCell = PianoCell(768, 256, note_train.embedding_len)
    pianoCell.train()
    optim = Adam(pianoCell.parameters(), lr=1e-3)
    pianoCell.to(device)
    pianoCell.train()
    mean_loss = 999.
    hc = None
    for i in range(epoch):
        epoch_loss = 0
        hc = pianoCell.init_hidden(note_train.batch_size) if hc is None else hc
        for j in range(step_num):
            print('epoch', i, 'step', j, end=' ')
            optim.zero_grad()
            note_inputs, off_inputs, note_labels, off_labels, mask = note_train.next()
            neg_labels = get_random_neg_labels(note_labels, note_train.embedding_len, 3000)
            neg_labels = torch.tensor(neg_labels, dtype=torch.long).to(device)
            mask = torch.tensor(mask, dtype=torch.float32).reshape(-1, 1).to(device)
            hc = (hc[0]*mask, hc[1]*mask)
            note_inputs = torch.tensor(note_inputs, dtype=torch.long).to(device)
            off_inputs = torch.tensor(off_inputs, dtype=torch.float32).to(device)
            note_labels = torch.tensor(note_labels, dtype=torch.long).to(device)
            off_labels = torch.tensor(off_labels, dtype=torch.float32).to(device)
            note_emb, off_reg, hc = pianoCell(note_inputs, off_inputs, hc)
            hc = (hc[0].detach(), hc[1].detach())
            cost = pianoCell.loss(note_emb, off_reg, note_labels, neg_labels, off_labels)
            cost.backward()
            optim.step()
            epoch_loss += cost.detach().cpu().numpy()
        epoch_loss /= step_num
        if epoch_loss<=mean_loss:
            mean_loss = epoch_loss
            torch.save(pianoCell.state_dict(), 'pianoCell_model.pth')
        print('epoch_loss:', epoch_loss, 'mean_loss:', mean_loss)

if __name__ == "__main__":
    train(100, 5000, 'raw_pieces.json')

