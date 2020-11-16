from Loss import PBLoss
from rnn_model import PianoBox
from note_process import Note
import torch
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, batch_size=32):
    note_data = Note('train_data.npz', batch_size)
    model = PianoBox(6, 256)
    optim = Adam(model.parameters(), lr=1e-4)
    loss = PBLoss()
    model.to(device)
    loss.to(device)
    model.train()
    mean_loss = 999
    for i in range(epoch):
        total_loss = 0
        step_cnt = 0
        for j in range(note_data.length//batch_size):
            print('epoch', i, 'step', j, end=' ')
            nf, gl, pl, cl, regl = note_data.next()
            optim.zero_grad()
            group_prob, pitch_prob, chord_prob, olv_vec = model(nf)
            cost = loss(group_prob, pitch_prob, chord_prob, olv_vec, gl, pl, cl, regl)
            cost.backward()
            optim.step()
            step_cnt += 1
            total_loss += cost.detach().cpu().numpy()
        total_loss = total_loss/step_cnt
        if total_loss<=mean_loss:
            mean_loss = total_loss
            torch.save(model.state_dict(), 'piano_model.pth')

if __name__ == "__main__":
    train(20)

            