from tqdm import tqdm
import os
from time import time
from torch.utils.data import DataLoader
from pathlib import Path
from deep_speaker_pytorch.utils import load_pickle,ensures_dir
from deep_speaker_pytorch.model import DeepSpeaker,Dense_Layer
from deep_speaker_pytorch.constants import EPOCHS,LEARNING_RATE,BATCH_SIZE,NUM_FRAMES
from deep_speaker_pytorch.batcher import DeepSpeakerDatasetSoftmax,LazyTripletBatcher
from deep_speaker_pytorch.triplet_loss import DeepSpeakerLoss
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim 
import torch.nn as nn   

# writer=SummaryWriter()

def train_softmax_model(working_dir:Path):
    train_dataset=DeepSpeakerDatasetSoftmax(os.path.join(working_dir,'keras-inputs','kx_train.npy'),
                                            os.path.join(working_dir,'keras-inputs','ky_train.npy'))
    test_dataset=DeepSpeakerDatasetSoftmax(os.path.join(working_dir,'keras-inputs','kx_test.npy'),
                                            os.path.join(working_dir,'keras-inputs','ky_test.npy'))
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # writer.add_text('device',f"{device}")

    spkrs=load_pickle(os.path.join(working_dir,'keras-inputs','categorical_speakers.pkl'))
    num_spkrs=len(spkrs.speaker_ids)

    dsm=DeepSpeaker()
    dense_layer=Dense_Layer(num_spkrs)

    # writer.add_graph(dsm)
    # writer.add_graph(dense_layer)
    dsm=dsm.to(device)
    dense_layer=dense_layer.to(device)
    optimizer=optim.Adam(dsm.parameters(),lr=LEARNING_RATE)
    criterion=nn.CrossEntropyLoss()
    ensures_dir(os.path.join(working_dir,'model_checkpoints','softmax'))
    for i in tqdm(range(EPOCHS),desc='epochs'):
        train_loss=0.0
        test_loss=0.0
        train_len=len(train_dataloader.dataset)
        test_len=len(test_dataloader.dataset)
        start=time()
        for data,target in tqdm(train_dataloader,desc='train'):
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=dsm(data)
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
        # writer.add_scaler('train_loss',train_loss,i)
        # writer.add_scaler('test_loss',test_loss,i)
        if (i+1)%10==0:
            torch.save(dsm,os.path.join(working_dir,'model_checkpoints','softmax',f"model_checkpoint_{i+1}.pt"))
        print(f"epoch {i+1} train_loss : {train_loss:.4f} test_loss : {test_loss:.4f} processing_time : {end-start:.4f}")

    torch.save(dsm,os.path.join(working_dir,'model_checkpoints','softmax',f"model_checkpoint_final.pt"))
    return dsm

def train_triplet_model(working_dir,
                        train_size=2000,
                        test_size=200,
                        lr=LEARNING_RATE):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DeepSpeaker()
    batcher=LazyTripletBatcher(working_dir,NUM_FRAMES,model,device)
    test_batches = []
    for _ in tqdm(range(test_size), desc='Build test set'):
        test_batches.append(batcher.get_batch_test(BATCH_SIZE))
    
    ensures_dir(os.path.join(working_dir,'model_checkpoints','triplet'))

    optimizer=optim.SGD(model.parameters(),lr=lr)
    criterion=DeepSpeakerLoss()
    model=model.to(device)
    for i in tqdm(range(EPOCHS),desc='epochs'):
        train_loss=0.0
        test_loss = 0.0
        train_len=train_size
        test_len=test_size
        start=time()
        train_batches=[]
        for _ in tqdm(range(train_size),desc="Building train set"):
            train_batches.append(batcher.get_random_batch(BATCH_SIZE, is_test=False))
        
        for data,target in tqdm(train_batches):
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(data.float())
            loss=criterion(target,output)+model.l2_regularization()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
        with torch.no_grad():
            for data,target in tqdm(test_batches):
                data=data.to(device)
                target=target.to(device)
                output=model(data.float())
                loss=criterion(target,output)+model.l2_regularization()
                test_loss+=loss.item()
        end=time()
        train_loss/=train_len
        test_loss/=test_len
        if (i+1)%10==0:
            torch.save(model,os.path.join(working_dir,'model_checkpoints','triplet',f'model_checkpoint_{i+1}.pt'))
        print(f"epoch {i+1} train_loss : {train_loss:.4f} test_loss : {test_loss:.4f} processing_time : {end-start:.4f}")
    torch.save(model,os.path.join(working_dir,'model_checkpoints','triplet',f"model_checkpoint_final.pt"))


if __name__=='__main__':
    # train_softmax_model('/mnt/d/Programs/Python/PW/projects/speech/embeddings/working_dir')
    train_triplet_model('/mnt/d/Programs/Python/PW/projects/speech/embeddings/working_dir')