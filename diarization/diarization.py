from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import tensorflow as tf
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.constants import NUM_FRAMES,SAMPLE_RATE
from deep_speaker.audio import mfcc_fbank
from deep_speaker.batcher import sample_from_mfcc
from tensorflow.keras import  models, layers,Model # type:ignore
import pandas as pd
import numpy as np
import json
import soundfile as sf
from io import BytesIO
import argparse
from time import perf_counter
import librosa
from python_speech_features import logfbank
from sklearn.cluster import AgglomerativeClustering # SpectralClustering,
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

DG_KEY = "deepgram api key"
deepgram=DeepgramClient(DG_KEY)

class CNNBiLSTMModel(Model):
    def __init__(self):
        super(CNNBiLSTMModel, self).__init__()
        self.conv_1=layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        self.max_pool_1=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.conv_2=layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        self.max_pool_2=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.conv_3=layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu',
                                                        kernel_initializer='glorot_uniform',
                                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        self.max_pool_3=layers.TimeDistributed(layers.MaxPooling2D((2,2)))
        self.flatten=layers.TimeDistributed(layers.Flatten())
        self.dense_1=layers.TimeDistributed(layers.Dense(128, activation='relu'))
        self.dropout_1=layers.Dropout(0.5)
        self.bilstm=layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dropout_2=layers.Dropout(0.5)
        self.dense_2=layers.TimeDistributed(layers.Dense(2, activation='softmax'))
        self.cnn_lstm=models.Sequential()
        self.cnn_lstm.add(layers.Input(shape=(None, 32, 32, 1)))
        self.cnn_lstm.add(self.conv_1)
        self.cnn_lstm.add(self.max_pool_1)
        self.cnn_lstm.add(self.conv_2)
        self.cnn_lstm.add(self.max_pool_2)
        self.cnn_lstm.add(self.conv_3)
        self.cnn_lstm.add(self.max_pool_3)
        self.cnn_lstm.add(self.flatten)
        self.cnn_lstm.add(self.dense_1)
        self.cnn_lstm.add(self.dropout_1)
        self.cnn_lstm.add(self.bilstm)
        self.cnn_lstm.add(self.dropout_2)
        self.cnn_lstm.add(self.dense_2)
    def call(self,x):
        x=self.cnn_lstm(x)
        return x
    

def get_timestamps(audio_path,model):
    wav=read_audio(audio_path)
    speech_timestamps=get_speech_timestamps(wav,model)
    start,end=[],[]
    for d in speech_timestamps:
        start.append(d['start'])
        end.append(d['end'])
    return start,end

def resize_audio_to_nearest_multiple(y, target_multiple):
    original_length = len(y)
    nearest_multiple = int(np.ceil(original_length / target_multiple) * target_multiple)
    if nearest_multiple > original_length:
        padding_length = nearest_multiple - original_length
        y_resized = np.pad(y, (0, padding_length), mode='constant')
    else:
        y_resized = y[:nearest_multiple]

    return y_resized

def pad_to_multiple_of_6(arr):
    batch_size = arr.shape[0]
    remainder = batch_size % 6
    if remainder != 0:
        padding_batches = 6 - remainder
        padding_shape = (padding_batches, *arr.shape[1:]) 
        padding_array = np.zeros(padding_shape, dtype=arr.dtype)
        arr_padded = np.concatenate((arr, padding_array), axis=0)
    else:
        arr_padded = arr 
    return arr_padded

def detect_speaker_change(arr,model):
    data=librosa.util.normalize(arr)
    original_size=data.shape[0]
    num_preds=np.ceil(original_size/5280)
    data=resize_audio_to_nearest_multiple(data,5280*6)
    start=0
    chunk_list=[]
    for end in range(5280,len(data)-5280,5280):
        chunk=data[start:end]
        m=logfbank(chunk,16000,nfilt=32).reshape(32,-1,1)
        start=end
        chunk_list.append(m.tolist())
    chunk_list=np.array(chunk_list)
    chunk_list=pad_to_multiple_of_6(chunk_list)
    chunk_list=chunk_list.reshape(-1,6,32,32,1)
    pred=model.predict(chunk_list,verbose=False)
    pred=pred.reshape(-1,2)
    overlaps=np.argmax(pred,axis=1)
    overlaps_idx=np.where(overlaps[:int(num_preds)]==1)
    return overlaps_idx[0]*5280

def re_chunk(dataframe:pd.DataFrame):
    change_list=dataframe['speaker_change'].to_list()
    rechunks={"start":[],'end':[],'arr':[]}
    remove_index=[]
    for i in range(len(change_list)):
        if len(change_list[i])>0:
            remove_index.append(i)
            start=df.iloc[i,0]
            splits=np.split(df.iloc[i,2],change_list[i])
            for split in splits:
                rechunks['arr'].append(split)
                rechunks['start'].append(start)
                start=start+len(split)
                rechunks['end'].append(start)
    if len(rechunks)>0:
        dataframe.drop(remove_index,axis=0,inplace=True)
        dataframe.drop(labels=['speaker_change'],axis=1,inplace=True)
        dataframe=pd.concat([dataframe,pd.DataFrame(rechunks)],ignore_index=True)
        dataframe.sort_values(by='start',ascending=True,inplace=True)
    return dataframe
    
def get_mfcc(arr,sampl_rate,num_frames):
    energy = np.abs(arr)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = arr[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sampl_rate)
    mfcc=sample_from_mfcc(mfcc,num_frames)
    return mfcc

def get_embedding(arr:np.ndarray,model:DeepSpeakerModel,sample_rate,num_frames):
    mfcc=get_mfcc(arr,sample_rate,num_frames)
    emb=model.m.predict(np.expand_dims(mfcc,axis=0),verbose=False)
    return emb.squeeze() 

def concatenate_rows(df):
    concatenated_rows = []
    current_row = df.iloc[0].copy()

    for i in range(1, len(df)):
        row = df.iloc[i]
        # if row['start'] == current_row['end'] and row['labels'] == current_row['labels']:
        if row['labels'] == current_row['labels']:
            current_row['arr'] = np.concatenate([current_row['arr'], row['arr']])
            current_row['end'] = row['end']
        else:
            concatenated_rows.append(current_row)
            current_row = row.copy()
    concatenated_rows.append(current_row)

    new_df = pd.DataFrame(concatenated_rows)
    return new_df

def get_deepgram_transcript(arr,sample_rate):
    audio_file=BytesIO()
    sf.write(audio_file,data=arr,samplerate=sample_rate,format='wav')
    audio_file.seek(0)
    payload: FileSource = {
        "buffer": audio_file,
    }
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        language='hi-Latn',
    )
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    json_data=json.loads(response.to_json(indent=4))
    transcript=""
    
    try:
        transcript =json_data.get('results').get('channels')[0].get('alternatives')[0].get('transcript')
        return transcript
    except Exception as _:
        return ""
def get_deepgram_file_transcript(arr:np.ndarray):
    audio_buffer=BytesIO()
    sf.write(audio_buffer,data=arr,samplerate=SAMPLE_RATE,format='wav')
    audio_buffer.seek(0)
    payload: FileSource = {
        "buffer": audio_buffer,
    }
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        language='hi-Latn',
        diarize=True
    )
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    json_data=json.loads(response.to_json(indent=4))
    return json_data

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--file_path",required=True)
    args=parser.parse_args()
    audio_path=args.file_path
    file_name=audio_path.split('/')[-1]
    silero_model = load_silero_vad()
    scd_model=CNNBiLSTMModel()
    scd_model.load_weights('/home/oriserve_ai/newhd/rahim_voice/voxseg/model_checkpoints/90-0.11.keras')
    emb_model=DeepSpeakerModel()
    emb_model.m.load_weights('/home/oriserve_ai/newhd/rahim_voice/deep-speaker/deep-speaker/checkpoints-triplets/ResCNN_checkpoint_2751.keras')
    start_time=perf_counter()
    start,end=get_timestamps(audio_path,silero_model)
    df=pd.DataFrame({'start':start,'end':end})
    data,sr=librosa.load(audio_path,sr=16000,mono=True)
    df['arr']=df.apply(lambda row:data[row['start']:row['end']],axis=1)
    df['speaker_change']=df['arr'].apply(lambda x:detect_speaker_change(x,scd_model))
    df=re_chunk(df)
    df=df[df['start']!=df['end']]
    df['embedding']=df['arr'].apply(lambda x:get_embedding(x,emb_model,SAMPLE_RATE,NUM_FRAMES))
    # spectral_model_rbf = SpectralClustering(n_clusters = 2, affinity ='rbf') 
    ems=df['embedding'].to_list()
    ems=np.array(ems)
    # labels_rbf = spectral_model_rbf.fit_predict(ems) 
    agglo_clustering=AgglomerativeClustering(n_clusters=2)
    labels_agglo=agglo_clustering.fit_predict(ems)
    # df['labels']=labels_rbf
    df['labels']=labels_agglo
    df=concatenate_rows(df)
    print(perf_counter()-start_time)
    df['transcript']=df.apply(lambda row:get_deepgram_transcript(data[row['start']:row['end']],SAMPLE_RATE),axis=1)
    df.drop(labels=['arr','embedding'],axis=1,inplace=True)
    df['start']=df['start']/16_000
    df['end']=df['end']/16_000
    df.to_csv(f'/home/oriserve_ai/newhd/rahim_voice/voxseg/transcriptions/our_method/{file_name.split('.')[0]}.csv',index=False)
    json_transcript=get_deepgram_file_transcript(data)
    sentences=json_transcript.get('results').get('channels')[0].get('alternatives')[0].get('paragraphs').get('paragraphs')
    with open(f'/home/oriserve_ai/newhd/rahim_voice/voxseg/transcriptions/deepgram/{file_name.split('.')[0]}.csv','a') as file:
        file.write('speaker,start,end,transcription\n')
        for sentence in sentences:
            text=sentence.get('sentences')[0].get('text')
            start=sentence.get('sentences')[0].get('start')
            end=sentence.get('sentences')[0].get('end')
            speaker=sentence.get('speaker')
            file.write(f'{speaker},{start},{end},{text}\n')
