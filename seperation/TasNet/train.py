import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import os,glob,random,yaml
import argparse
import librosa
import soundfile as sf  
import numpy as np
from itertools import permutations
from tqdm import tqdm
from time import perf_counter

class Encoder(nn.Module):
    def __init__(self,L,N):
        super(Encoder,self).__init__()
        """
        L: Number of input channels(number of samples per segment)
        N: Number of output channels(number of basis signals)
        """
        self.L = L
        self.N = N
        self.EPS = 1e-8
        self.conv1d_U=nn.Conv1d(in_channels=L,out_channels=N,kernel_size=1,stride=1,bias=False)
        self.conv1d_V=nn.Conv1d(in_channels=L,out_channels=N,kernel_size=1,stride=1,bias=False)
    
    def forward(self,mixture):
        """
        mixture:Tensor of shape [B,K,L] where K are the number of segment being processed at once
        output: Tensor of shape [B,K,N] where N are the number of basis signals
        """
        B,K,L=mixture.size()
        norm_coef=torch.norm(mixture,p=2,dim=2,keepdim=True)
        normed_mixture=mixture/(norm_coef+self.EPS)
        normed_mixture=torch.unsqueeze(normed_mixture.view(-1,L),2)
        conv=F.relu(self.conv1d_U(normed_mixture))
        gate=F.sigmoid(self.conv1d_V(normed_mixture))
        mixture_w=conv*gate
        mixture_w=mixture_w.view(B,K,self.N)
        return mixture_w,norm_coef

class Separator(nn.Module):
    def __init__(self,N:int,hidden_size,num_layers,bidirectional=False,nspk=2) -> None:
        super(Separator,self).__init__()
        self.N=N
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.num_layers=num_layers
        self.nspk=nspk
        self.layer_norm=nn.LayerNorm(N)
        self.LSTM=nn.LSTM(input_size=N,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,batch_first=True)
        fc_in_dim=hidden_size*2 if bidirectional else hidden_size
        self.fc=nn.Linear(fc_in_dim,nspk*N)
    
    def forward(self,mixture_w):
        """
        mixture_w: Tensor of shape [B,K,N]
        output: Tensor of shape [B,K,nspk,N]
        """
        B,K,N=mixture_w.size()
        normed_mixture_w=self.layer_norm(mixture_w)
        output,_=self.LSTM(normed_mixture_w)
        score=self.fc(output)
        score=score.view(B,K,self.nspk,N)
        est_mask=F.softmax(score,dim=2)
        return est_mask

class Decoder(nn.Module):
    def __init__(self,N,L):
        super(Decoder,self).__init__()
        self.N=N
        self.L=L
        self.basis_signals=nn.Linear(N,L,bias=False)
    
    def forward(self,mixture_w,est_mask,norm_coef):
        """
        mixture_w: Tensor of shape [B,K,N]
        est_mask: Tensor of shape [B,K,nspk,N]
        norm_coef: Tensor of shape [B,K,1]
        output: Tensor of shape [B,nspk,K,L]
        """
        source_w=torch.unsqueeze(mixture_w,2)*est_mask
        est_source=self.basis_signals(source_w)
        norm_coef=torch.unsqueeze(norm_coef,2)
        est_source=est_source*norm_coef
        est_source=est_source.permute(0,2,1,3).contiguous()
        return est_source

class TasNet(nn.Module):
    def __init__(self,L,N,hidden_size,num_layers,bidirectional=False,nspk=2):
        super(TasNet,self).__init__()
        self.L=L
        self.N=N
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.nspk=nspk
        self.encoder=Encoder(L,N)
        self.separator=Separator(N,hidden_size,num_layers,bidirectional,nspk)
        self.decoder=Decoder(N,L)
    
    def forward(self,mixture):
        mixture_w,norm_coef=self.encoder(mixture)
        est_mask=self.separator(mixture_w)
        est_source=self.decoder(mixture_w,est_mask,norm_coef)
        return est_source

class AudioDataset(Dataset):
    def __init__(self,L:int,K:int,folder_path:str,sample_rate=8000) -> None:
        self.L=L
        self.K=K
        self.folder_path=folder_path
        self.sample_rate=sample_rate
        self.files=glob.glob(os.path.join(folder_path,'*.wav'))
        self.audio_info=self.load_audio_info()
    
    def __len__(self):
        return len(self.audio_info['path'])
    
    def __getitem__(self,idx):
        audio_path=self.audio_info['path'][idx]
        start=self.audio_info['start'][idx]/self.sample_rate
        end=self.audio_info['end'][idx]/self.sample_rate
        audio1,_=librosa.load(audio_path,sr=self.sample_rate,mono=True,offset=start,duration=end-start)
        # load a random audio from the data
        i=random.randint(a=0,b=len(self.audio_info['path'])-2)
        while i==idx:
            i=random.randint(a=0,b=len(self.audio_info['path'])-2)
        
        audio_path=self.audio_info['path'][i]
        start=self.audio_info['start'][i]/self.sample_rate
        end=self.audio_info['end'][i]/self.sample_rate
        audio2,_=librosa.load(audio_path,sr=self.sample_rate,mono=True,offset=start,duration=end-start)
        mixture=audio1+audio2
        mixture=librosa.util.normalize(mixture)
        audio1=librosa.util.normalize(audio1)
        audio2=librosa.util.normalize(audio2)
        mixture=torch.from_numpy(mixture.reshape(self.K,self.L))
        sources=torch.from_numpy(np.array([audio1.reshape(self.K,self.L),audio2.reshape(self.K,self.L)]))
        return mixture,sources

    def load_audio_info(self):
        audio_info=dict(path=list(),start=list(),end=list())
        for file in self.files:
            info=sf.info(os.path.join(self.folder_path,file))
            duration=int(info.duration*self.sample_rate)
            chunk_length=self.L*self.K
            start=0
            for i in range(chunk_length,duration,chunk_length):
                if(i-start)==chunk_length:
                    audio_info['path'].append(info.name)
                    audio_info['start'].append(start)
                    audio_info['end'].append(i)
                    start=i
        return audio_info

# SI-SNR with PIT(Permutation Invariant Training)
def calculate_si_snr(source:torch.tensor,estimate:torch.tensor,eps=1e-8):
    """
    source: Tensor of shape [B,C,K,L]
    estimate: Tensor of shape [B,C,K,L]
    eps: small value to avoid division by zero
    output: Tensor of shape [B,K]
    B: Batch size
    C: Number of speakers
    K: Number of segments
    L: Number of samples per segment
    T=K*L
    """
    B,C,K,L=source.size()
    flat_source=source.view(B,C,-1) # [B,C,T]
    flat_estimate=estimate.view(B,C,-1) # [B,C,T]
    s_target=torch.unsqueeze(flat_source,dim=1) # [B,1,C,T]
    s_estimate=torch.unsqueeze(flat_estimate,dim=2) # [B,C,1,T]
    pair_wise_dot=torch.sum(s_target*s_estimate,dim=3,keepdim=True) # [B,C,C,1]
    s_target_energy=torch.sum(s_target**2,dim=3,keepdim=True)+eps # [B,1,C,1]
    pairwise_proj=pair_wise_dot*s_target/s_target_energy # [B,C,C,T]
    e_noise=s_estimate-pairwise_proj # [B,C,C,T]
    pair_wise_si_snr=torch.sum(pairwise_proj**2,dim=3)/(torch.sum(e_noise**2,dim=3)+eps) # [B,C,C]
    pair_wise_si_snr=10*torch.log10(pair_wise_si_snr+eps) # [B,C,C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    loss=0-torch.mean(max_snr)
    return loss

def train(model,
          device,
          train_loader,
          optimizer,
          epochs,
          save_path):
    model.train()
    for epoch in tqdm(range(epochs),desc="Epochs"):
        start=perf_counter()
        train_loss=0
        train_len=len(train_loader.dataset)
        for mixture,sources in tqdm(train_loader,desc="Training"):
            mixture=mixture.to(device)
            sources=sources.to(device)
            # print(f"mixture_shape: {mixture.shape}")
            # print(f"sources_shape: {sources.shape}")
            est_sources=model(mixture)
            optimizer.zero_grad()
            # print(f"est_sources_shape: {est_sources.shape}")
            si_snr=calculate_si_snr(sources,est_sources)
            # print(f"si_snr_shape: {si_snr.item()}")
            train_loss+=si_snr.item()
            si_snr.backward()
            optimizer.step()
        end=perf_counter()
        print(f"Epoch: {epoch+1}/{epoch} Loss: {train_loss/train_len} Time: {end-start}")
    torch.save(model,save_path+"/model.pt")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True)
    args=parser.parse_args()
    with open(args.config) as f:
        config=yaml.safe_load(f)
    # print(config)
    L=config['model']['L']
    N=config['model']['N']
    K=config['dataset']['K']
    sample_rate=config['dataset']['sample_rate']
    audio_folder=config['dataset']['folder_path']
    hidden_size=config['model']['hidden_size']
    num_layers=config['model']['num_layers']
    bidirectional=config['model']['bidirectional']
    nspk=config['model']['nspk']
    epochs=config['training']['epochs']
    batch_size=config['training']['batch_size']
    lr=config['training']['lr']
    save_path=config['training']['save_folder']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TasNet(L=L,N=N,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,nspk=nspk).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    train_loader=DataLoader(AudioDataset(L=L,K=K,folder_path=audio_folder,sample_rate=sample_rate),batch_size=batch_size,shuffle=True)
    train(model,device,train_loader,optimizer,epochs,save_path)