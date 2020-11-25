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
        self.reg_loss = nn.MSELoss()
    
    def forward(self, pitch_prob, olv_vec, pitch_labels, olv_labels):
        pitch_loss = self.cosloss(pitch_prob, pitch_labels.long())
        olv_loss = self.reg_loss(olv_vec, olv_labels.float())
        total_loss = pitch_loss+olv_loss
        print('total-loss:', total_loss.detach().cpu().numpy(), 'p-loss:', pitch_loss.detach().cpu().numpy(), 'olv-loss:', olv_loss.detach().cpu().numpy())
        return total_loss

if __name__ == "__main__":
    pass