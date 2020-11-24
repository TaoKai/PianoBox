from Loss import PBLoss
from rnn_model import PianoBox
from note_process import Note
import torch
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, batch_size=32):
    model_path = 'piano_model.pth'
    note_data = Note('raw_pieces.json', batch_size)
    model = PianoBox(512, note_data.note_num, note_data.offset_num)
    optim = Adam(model.parameters(), lr=3e-5)
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
            p, o, pl, ol = note_data.next()
            optim.zero_grad()
            pitch_prob, olv_vec, _ = model(p, o)
            cost = loss(pitch_prob, olv_vec, pl, ol)
            cost.backward()
            optim.step()
            step_cnt += 1
            total_loss += cost.detach().cpu().numpy()
        total_loss = total_loss/step_cnt
        if total_loss<=mean_loss:
            mean_loss = total_loss
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train(10000, 128)

            