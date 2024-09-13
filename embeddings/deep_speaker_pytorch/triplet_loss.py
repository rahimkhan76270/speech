import torch
import torch.nn as nn   
import torch.nn.functional as F 
from collections import deque

class DIEMSimilarity(nn.Module):
    def __init__(self,v_m=-1,v_M=1):
        super(DIEMSimilarity,self).__init__()
        self.dq=deque(maxlen=1000)
        self.v_M=v_M
        self.v_m=v_m
        self.ed=None
        self.var=None
    
    def forward(self,batch1:torch.tensor,batch2:torch.tensor)->torch.tensor:
        for v1,v2 in zip(batch1,batch2):
            self.dq.append(v1.tolist())
            self.dq.append(v2.tolist())
        t=torch.tensor(list(self.dq))
        self.var=t.var()
        pdist=F.pdist(t)
        self.ed=pdist.mean()
        # self.v_M=t.max()
        # self.v_m=t.min()
        batch_diem=(self.v_M-self.v_m)*(torch.linalg.vector_norm(batch1-batch2,ord=2,axis=1)-self.ed)/self.var
        return batch_diem

def batch_cosine_similarity(x,y):
    x_norm=torch.norm(x,dim=1,keepdim=True)
    y_norm=torch.norm(y,dim=1,keepdim=True)
    return torch.matmul(x,y.T)/x_norm/y_norm

class DeepSpeakerLoss(nn.Module):
  def __init__(self) -> None:
    super(DeepSpeakerLoss,self).__init__()
    self.alpha=0.1
    self.diem=DIEMSimilarity()
  
  def forward(self,y_true,y_pred):
    split=y_pred.shape[0]//3
    anchor=y_pred[:split]
    positive=y_pred[split:2*split]
    negative=y_pred[2*split:]
    # cos_sim_ap=batch_cosine_similarity(anchor,positive)
    # cos_sim_an=batch_cosine_similarity(anchor,negative)
    diem_sim_ap=self.diem(anchor,positive)
    diem_sim_an=self.diem(anchor,negative)
    # loss=torch.max(cos_sim_an-cos_sim_ap+self.alpha,torch.Tensor(0.0))
    loss=torch.max(diem_sim_ap-diem_sim_an+self.alpha,torch.tensor(0.0))
    return torch.mean(loss)
