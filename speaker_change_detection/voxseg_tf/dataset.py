import os
import numpy as np
from glob import glob
import librosa
from python_speech_features import logfbank  # type:ignore


class DataGenerator:
    def __init__(self,folder_path
                 ,time_dim=6,
                 data_len=10000,
                 num_frames=5280,
                 file_ext='wav',
                 sample_rate=16_000,
                 nfilt=32):
        self.time_dim=time_dim
        self.folder_path=folder_path
        self.data_len=data_len
        self.sample_rate=sample_rate
        self.nfilt=nfilt
        self.speakers=os.listdir(folder_path)
        self.num_frames=num_frames
        self.speaker_file_dict={}
        for speaker in self.speakers:
            self.speaker_file_dict[speaker]=glob(os.path.join(folder_path,speaker,f'*.{file_ext}'))
    def __getitem__(self,idx):
        return self.make_data()

    def generator(self):
        for _ in range(self.data_len):
            yield self.make_data()
    def make_data(self):
        choiced_speakers=np.random.choice(self.speakers,size=self.time_dim,replace=False)
        data=[]
        label=[]
        for _ in range(self.time_dim):
            voice_change=True if np.random.rand()>0.5 else False
            if voice_change:
                np.random.shuffle(choiced_speakers)
                spk1=choiced_speakers[0]
                spk2=choiced_speakers[1]
                file1=np.random.choice(self.speaker_file_dict[spk1],size=1)[0]
                file2=np.random.choice(self.speaker_file_dict[spk2],size=1)[0]
                y1,sr1=librosa.load(file1,mono=True,sr=self.sample_rate)
                y2,sr2=librosa.load(file2,mono=True,sr=self.sample_rate)
                starting_point_y1=np.random.randint(0,len(y1)-self.num_frames)
                starting_point_y2=np.random.randint(0,len(y2)-self.num_frames)
                random_between_point=np.random.randint(200,self.num_frames-self.num_frames//2)
                y1=y1[starting_point_y1:starting_point_y1+random_between_point]
                y2=y2[starting_point_y2:starting_point_y2+self.num_frames-random_between_point]
                y=np.concatenate([y1,y2])
                y=librosa.util.normalize(y)
                data.append(logfbank(y,self.sample_rate,nfilt=self.nfilt).reshape(self.nfilt,-1,1))
                label.append([0,1])
            else:
                np.random.shuffle(choiced_speakers)
                spk=choiced_speakers[0]
                file=np.random.choice(self.speaker_file_dict[spk],size=1)[0]
                y,sr=librosa.load(file,mono=True,sr=self.sample_rate)
                y=librosa.util.normalize(y)
                starting_point=np.random.randint(0,len(y)-self.num_frames)
                end_point=starting_point+self.num_frames
                y=y[starting_point:end_point]
                data.append(logfbank(y,self.sample_rate,nfilt=self.nfilt).reshape(self.nfilt,-1,1))
                label.append([1,0])
        return np.array(data),np.array(label)
