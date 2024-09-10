import dill
import os
from glob import glob
import logging
import numpy as np  
from random import choice

logger=logging.getLogger(__name__)

def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def find_files(directory, ext='wav'):
    return sorted(glob(directory + f'/**/*.{ext}', recursive=True))

def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)



def load_pickle(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading PKL file: {file}.')
    with open(file, 'rb') as r:
        return dill.load(r)


def load_npy(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading NPY file: {file}.')
    return np.load(file)


def train_test_sp_to_utt(audio, is_test):
    sp_to_utt = {}
    for speaker_id, utterances in audio.speakers_to_utterances.items():
        utterances_files = sorted(utterances.values())
        train_test_sep = int(len(utterances_files) * 0.8)
        sp_to_utt[speaker_id] = utterances_files[train_test_sep:] if is_test else utterances_files[:train_test_sep]
    return sp_to_utt

def extract_speaker(utt_file):
    return utt_file.split('/')[-1].split('_')[0]


def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)


def sample_from_mfcc_file(utterance_file, max_length):
    mfcc = np.load(utterance_file)
    return sample_from_mfcc(mfcc, max_length)
