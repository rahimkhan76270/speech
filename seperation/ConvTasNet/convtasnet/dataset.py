import tensorflow as tf 
from glob import glob
import soundfile as sf
import librosa
import numpy as np  
from tqdm import tqdm

class ConvTasNetDataGenerator:
    """
    audio sample rate should be same as passed in the class there is no auto conversion done in this class
    """
    def __init__(self,folder_path:str,
                audio_chunk_len=2,
                num_spks=2,
                sample_rate=8_000,
                data_len=1000,
                file_ext='wav'):
        self.all_files=glob(f"{folder_path}/*.{file_ext}")
        self.sample_rate=sample_rate
        self.num_spks=num_spks
        self.audio_chunk_len=audio_chunk_len
        self.audio_with_start_end=self.load_files()
        self.data_len=data_len
        self.folder_path=folder_path
        self.file_ext=file_ext

    def load_files(self):
        files=[]
        for file in tqdm(self.all_files):
            info=sf.info(file)
            duration=info.duration
            assert info.samplerate == self.sample_rate, f"sample rate should be {self.sample_rate} but found {info.samplerate} in the file {file}"
            start=0
            for end in range(int(self.sample_rate*self.audio_chunk_len),int(duration*self.sample_rate),int(self.sample_rate*self.audio_chunk_len)):
                if not  end-start<self.audio_chunk_len*self.sample_rate:
                    files.append({"path":file,"start":start,'end':end})
                start=end
        
        # assert len(files)==0, f"no files found in the folder {self.folder_path} with file ext. {self.file_ext}"
        print(f"{len(files)} files loaded")
        return files

    def __len__(self):
        return len(self.audio_with_start_end)

    def __getitem__(self,idx):
        files=np.random.choice(self.audio_with_start_end,self.num_spks,replace=False)
        x=0
        y=[]
        for file in files:
            path=file['path']
            start=file['start']/self.sample_rate
            end=file['end']/self.sample_rate
            data,_=librosa.load(path,mono=True,offset=start,duration=end-start,sr=self.sample_rate,dtype='float32')
            x=x+data
            y.append(data.tolist())
        x=tf.constant(np.expand_dims(x,axis=0))
        y=tf.constant(np.expand_dims(y,axis=1))
        return x,y

    def generator(self):
        for idx in range(self.data_len):
            yield self[idx]
