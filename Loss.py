import torch
import torch.nn as nn
import torch.nn.functional as F
is_cuda = torch.cuda.is_available()
class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1])
        if is_cuda:
            self.y = self.y.cuda()
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        
        return cosine_loss + self.xent * cent_loss

class PBLoss(nn.Module):
    def __init__(self):
        super(PBLoss, self).__init__()
        self.cosloss = CosineLoss()
        self.smoothl1 = nn.SmoothL1Loss()
    
    def forward(self, group_prob, pitch_prob, chord_prob, olv_vec, group_labels, pitch_labels, chord_labels, olv_labels):
        group_loss = self.cosloss(group_prob, group_labels.long())
        pitch_loss = self.cosloss(pitch_prob, pitch_labels.long())
        chord_loss = self.cosloss(chord_prob, chord_labels.long())
        olv_loss = self.smoothl1(olv_vec, olv_labels)
        total_loss = group_loss+pitch_loss+chord_loss+olv_loss
        print('total-loss:', total_loss.detach().cpu().numpy(), 'g-loss:', group_loss.detach().cpu().numpy(), 'p-loss:', pitch_loss.detach().cpu().numpy(), 'c-loss:', chord_loss.detach().cpu().numpy(), 'olv-loss:', olv_loss.detach().cpu().numpy())
        return total_loss

if __name__ == "__main__":
    from rnn_model import PianoBox
    from note_process import Note
    PB = PianoBox(6, 128)
    note = Note('train_data.npz', 16)
    nf, gl, pl, cl, regl = note.next()
    group_prob, pitch_prob, chord_prob, olv_vec = PB(nf)
    pbloss = PBLoss()
    cost = pbloss(group_prob, pitch_prob, chord_prob, olv_vec, gl, pl, cl, regl)