import torch
import torch.nn as nn
import torch.nn.init as init 

class ClippedRelu(nn.Module):
  def __init__(self):
    super(ClippedRelu, self).__init__()
  def forward(self,x):
    return torch.clamp(x,min=0.0,max=20.0)
  

class IdentityBlock(nn.Module):
  def __init__(self,in_channels,kernel_size,filters,l2=0.0001):
    super(IdentityBlock, self).__init__()
    self.kernel_size = kernel_size
    self.filters = filters
    self.l2=l2
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=1
                            ,bias=True,
                            padding="same")
    init.xavier_uniform_(self.conv1.weight)
    self.bn1 = nn.BatchNorm2d(num_features=filters)
    self.clipped_relu1 = ClippedRelu()
    self.conv2 = nn.Conv2d(in_channels=filters,
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=1
                            ,bias=True,
                            padding="same")
    init.xavier_uniform_(self.conv2.weight)
    self.bn2 = nn.BatchNorm2d(num_features=filters)
    self.clipped_relu2=ClippedRelu()
    self.clipped_relu3=ClippedRelu()

  def forward(self,x):
    x_shortcut = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.clipped_relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.clipped_relu2(x)
    x = x + x_shortcut
    return self.clipped_relu3(x)
  
  def l2_regularization(self):
    n=torch.norm(self.conv1.weight,p=2)+torch.norm(self.conv2.weight,p=2)
    return n*self.l2
  

class ConvResBlock(nn.Module):
  def __init__(self,in_channels,filters,l2=0.0001) -> None:
    super(ConvResBlock,self).__init__()
    self.filters=filters
    self.l2=l2
    self.conv1=nn.Conv2d(in_channels=in_channels
                        ,out_channels=filters,
                        kernel_size=5
                        ,stride=2
                        ,bias=True
                        ,padding=2)
    init.xavier_uniform_(self.conv1.weight)
    self.bn1=nn.BatchNorm2d(num_features=filters)
    self.clipped_relu1=ClippedRelu()
    self.id_block1=IdentityBlock(in_channels=filters,kernel_size=3,filters=filters,l2=l2)
    self.id_block2=IdentityBlock(in_channels=filters,kernel_size=3,filters=filters,l2=l2)
    self.id_block3=IdentityBlock(in_channels=filters,kernel_size=3,filters=filters,l2=l2)

  def forward(self,x):
    x=self.conv1(x)
    x=self.bn1(x)
    x=self.clipped_relu1(x)
    x=self.id_block1(x)
    x=self.id_block2(x)
    x=self.id_block3(x)
    return x
  def l2_regularization(self):
    n=torch.norm(self.conv1.weight,p=2)*self.l2+self.id_block1.l2_regularization()+self.id_block2.l2_regularization()+self.id_block3.l2_regularization()
    return n

class DeepSpeaker(nn.Module):
  def __init__(self,l2=0.0001) -> None:
    super(DeepSpeaker,self).__init__()
    self.l2=l2
    self.conv_res_block1=ConvResBlock(1,filters=64,l2=l2)
    self.conv_res_block2=ConvResBlock(64,filters=128,l2=l2)
#     self.conv_res_block3=ConvResBlock(128,filters=256,l2=l2)
#     self.conv_res_block4=ConvResBlock(256,filters=512,l2=l2)
    self.linear=nn.Linear(in_features=2048,out_features=512)

  def forward(self,x):
    x=self.conv_res_block1(x)
    x=self.conv_res_block2(x)
#     x=self.conv_res_block3(x)
#     x=self.conv_res_block4(x)
    batch,channels,height,width=x.shape
    x=torch.reshape(x,(batch,-1,2048))
    x=torch.mean(x,dim=1)
    x=self.linear(x)
    norm=torch.norm(x,p=2,dim=1,keepdim=True)
    return x/norm

  def l2_regularization(self):
    n=self.conv_res_block1.l2_regularization()+self.conv_res_block2.l2_regularization()
    return n

def Dense_Layer(num_spkrs):
  dense_layer=nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features=512,out_features=num_spkrs))
  return dense_layer
                            