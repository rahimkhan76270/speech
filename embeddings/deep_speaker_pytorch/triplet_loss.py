import torch
import torch.nn as nn   

def batch_cosine_similarity(x,y):
    x_norm=torch.norm(x,dim=1,keepdim=True)
    y_norm=torch.norm(y,dim=1,keepdim=True)
    return torch.matmul(x,y.T)/x_norm/y_norm

class DeepSpeakerLoss(nn.Module):
  def __init__(self) -> None:
    super(DeepSpeakerLoss,self).__init__()
    self.alpha=0.1
  
  def forward(self,y_true,y_pred):
    split=y_pred.shape[0]//3
    anchor=y_pred[:split]
    positive=y_pred[split:2*split]
    negative=y_pred[2*split:]
    cos_sim_ap=batch_cosine_similarity(anchor,positive)
    cos_sim_an=batch_cosine_similarity(anchor,negative)
    loss=torch.max(cos_sim_an-cos_sim_ap+self.alpha,torch.tensor(0.0))
    return torch.mean(loss)
