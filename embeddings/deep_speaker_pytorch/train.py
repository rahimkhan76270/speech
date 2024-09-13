from tqdm import tqdm
import os
from time import time
from torch.utils.data import DataLoader
from pathlib import Path
from deep_speaker_pytorch.utils import load_pickle,ensures_dir
from deep_speaker_pytorch.model import DeepSpeaker,Dense_Layer
from deep_speaker_pytorch.constants import EPOCHS,LEARNING_RATE,BATCH_SIZE,NUM_FRAMES
from deep_speaker_pytorch.batcher import DeepSpeakerDatasetSoftmax,LazyTripletBatcher,DeepSpeakerTripletDataset
from deep_speaker_pytorch.triplet_loss import DeepSpeakerLoss
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim 
import torch.nn as nn   

writer=SummaryWriter()

def train_softmax_model(working_dir:Path):
    train_dataset=DeepSpeakerDatasetSoftmax(os.path.join(working_dir,'keras-inputs','kx_train.npy'),
                                            os.path.join(working_dir,'keras-inputs','ky_train.npy'))
    test_dataset=DeepSpeakerDatasetSoftmax(os.path.join(working_dir,'keras-inputs','kx_test.npy'),
                                            os.path.join(working_dir,'keras-inputs','ky_test.npy'))
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer.add_text('device',f"{device}")

    spkrs=load_pickle(os.path.join(working_dir,'keras-inputs','categorical_speakers.pkl'))
    num_spkrs=len(spkrs.speaker_ids)

    dsm=DeepSpeaker()
    dense_layer=Dense_Layer(num_spkrs)

    writer.add_text('device',f"{device}")
    writer.add_text("learning_rate", f"{LEARNING_RATE}")
    writer.add_text("epochs",f"{EPOCHS}")
    writer.add_text("batch_size",f"{BATCH_SIZE}")
    
    dsm=dsm.to(device)
    dense_layer=dense_layer.to(device)
    optimizer=optim.Adam(dsm.parameters(),lr=LEARNING_RATE)
    criterion=nn.CrossEntropyLoss()
    ensures_dir(os.path.join(working_dir,'model_checkpoints','softmax'))
    dsm_graph_added=False
    dense_graph_added=False
    for i in tqdm(range(EPOCHS),desc='epochs'):
        dsm.train()
        train_loss=0.0
        test_loss=0.0
        train_len=len(train_dataloader.dataset)
        test_len=len(test_dataloader.dataset)
        start=time()
        for data,target in tqdm(train_dataloader,desc='train'):
            data=data.to(device)
            if not dsm_graph_added:
                writer.add_graph(dsm,input_to_model=data,verbose=False)
                dsm_graph_added=True
            target=target.to(device)
            optimizer.zero_grad()
            output=dsm(data)
            if not dense_graph_added:
                writer.add_graph(dense_layer,input_to_model=output,verbose=False)
                dense_graph_added=True
            output=dense_layer(output)
            loss=criterion(output,target.squeeze())+dsm.l2_regularization()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
        with torch.no_grad():
            for data,target in tqdm(test_dataloader,desc="test "):
                data=data.to(device)
                target=target.to(device)
                output=dsm(data)
                output=dense_layer(output)
                loss=criterion(output,target.squeeze())+dsm.l2_regularization()
                test_loss+=loss.item()
        end=time()
        train_loss/=train_len
        test_loss/=test_len
        writer.add_scalar('train_loss',train_loss,i)
        writer.add_scalar('test_loss',test_loss,i)
        if (i+1)%10==0:
            torch.save(dsm,os.path.join(working_dir,'model_checkpoints','softmax',f"model_checkpoint_{i+1}.pt"))
        print(f"epoch {i+1} train_loss : {train_loss:.4f} test_loss : {test_loss:.4f} processing_time : {end-start:.4f}")

    torch.save(dsm,os.path.join(working_dir,'model_checkpoints','softmax',f"model_checkpoint_final.pt"))
    return dsm

def train_triplet_model(working_dir,
                        train_size=200,
                        test_size=20,
                        lr=LEARNING_RATE):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DeepSpeaker()
    batcher=LazyTripletBatcher(working_dir,NUM_FRAMES,model,device)
    test_dataset=DeepSpeakerTripletDataset(batcher=batcher,num_batches=test_size,batch_size=BATCH_SIZE,is_test=True)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=None,shuffle=False)
    
    ensures_dir(os.path.join(working_dir,'model_checkpoints','triplet'))
    writer.add_text('device',f"using device {device}")
    writer.add_text("train_size",f"{train_size}")
    writer.add_text('test_size',f'{test_size}')
    writer.add_text("learning_rate" ,f"{LEARNING_RATE}")
    writer.add_text("epochs",f"{EPOCHS}")
    writer.add_text("batch_size",f"{BATCH_SIZE}")

    optimizer=optim.Adam(model.parameters(),lr=lr)
    criterion=DeepSpeakerLoss()
    model=model.to(device)
    graph_added=False
    for i in tqdm(range(EPOCHS),desc='epochs'):
        model.train()
        train_loss=0.0
        test_loss = 0.0
        train_len=train_size
        test_len=test_size
        train_dataset=DeepSpeakerTripletDataset(batcher=batcher,num_batches=train_size,batch_size=BATCH_SIZE,is_test=False)
        train_dataloader=DataLoader(train_dataset,batch_size=None,shuffle=False)
        start=time()
        for data,target in tqdm(train_dataloader,desc='train',total=train_size):
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            if not graph_added:
                writer.add_graph(model,input_to_model=data,verbose=False)
                graph_added=True
            output=model(data.float())
            loss=criterion(target,output)+model.l2_regularization()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
        with torch.no_grad():
            for data,target in tqdm(test_dataloader,desc='test',total=test_size):
                data=data.to(device)
                target=target.to(device)
                output=model(data.float())
                loss=criterion(target,output)+model.l2_regularization()
                test_loss+=loss.item()
        end=time()
        train_loss/=train_len
        test_loss/=test_len
        writer.add_scalar('loss/train',train_loss,i)
        writer.add_scalar('loss/test',test_loss,i)
        if (i+1)%10==0:
            torch.save(model,os.path.join(working_dir,'model_checkpoints','triplet',f'model_checkpoint_{i+1}.pt'))
        print(f"epoch {i+1} train_loss : {train_loss:.4f} test_loss : {test_loss:.4f} processing_time : {end-start:.4f}")
    torch.save(model,os.path.join(working_dir,'model_checkpoints','triplet',f"model_checkpoint_final.pt"))


if __name__=='__main__':
    # train_softmax_model('/media/ori_quadro/newhd1/rahim-voice/deep-speaker/working_dir')
    train_triplet_model('/mnt/d/Programs/Python/PW/projects/speech/embeddings/working_dir')
    writer.flush()
    writer.close()