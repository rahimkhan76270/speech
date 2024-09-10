import os
import logging
import dill
import numpy as np  
from tqdm import tqdm
import json
import torch
from torch.utils.data import Dataset
from collections import Counter,deque
from deep_speaker_pytorch.constants import NUM_FBANKS,NUM_FRAMES
from deep_speaker_pytorch.utils import load_npy,load_pickle,ensures_dir,sample_from_mfcc_file,train_test_sp_to_utt,extract_speaker
from deep_speaker_pytorch.triplet_loss import batch_cosine_similarity
from deep_speaker_pytorch.audio import Audio
from deep_speaker_pytorch.model import DeepSpeaker

logger=logging.getLogger(__name__)


class KerasFormatConverter:

    def __init__(self, working_dir, load_test_only=False):
        self.working_dir = working_dir
        self.output_dir = os.path.join(self.working_dir, 'keras-inputs')
        ensures_dir(self.output_dir)
        self.categorical_speakers = load_pickle(os.path.join(self.output_dir, 'categorical_speakers.pkl'))
        if not load_test_only:
            self.kx_train = load_npy(os.path.join(self.output_dir, 'kx_train.npy'))
            self.ky_train = load_npy(os.path.join(self.output_dir, 'ky_train.npy'))
        self.kx_test = load_npy(os.path.join(self.output_dir, 'kx_test.npy'))
        self.ky_test = load_npy(os.path.join(self.output_dir, 'ky_test.npy'))
        self.audio = Audio(cache_dir=self.working_dir, audio_dir=None)
        if self.categorical_speakers is None:
            self.categorical_speakers = SparseCategoricalSpeakers(self.audio.speaker_ids)

    def persist_to_disk(self):
        with open(os.path.join(self.output_dir, 'categorical_speakers.pkl'), 'wb') as w:
            dill.dump(self.categorical_speakers, w)
        np.save(os.path.join(self.output_dir, 'kx_train.npy'), self.kx_train)
        np.save(os.path.join(self.output_dir, 'kx_test.npy'), self.kx_test)
        np.save(os.path.join(self.output_dir, 'ky_train.npy'), self.ky_train)
        np.save(os.path.join(self.output_dir, 'ky_test.npy'), self.ky_test)

    def generate_per_phase(self, max_length=NUM_FRAMES, num_per_speaker=3000, is_test=False):
        # train OR test.
        num_speakers = len(self.audio.speaker_ids)
        sp_to_utt = train_test_sp_to_utt(self.audio, is_test)

        # 64 fbanks 1 channel(s).
        # float32
        kx = np.zeros((num_speakers * num_per_speaker, max_length, NUM_FBANKS, 1), dtype=np.float32)
        ky = np.zeros((num_speakers * num_per_speaker, 1), dtype=np.float32)

        desc = f'Converting to Keras format [{"test" if is_test else "train"}]'
        for i, speaker_id in enumerate(tqdm(self.audio.speaker_ids, desc=desc)):
            utterances_files = sp_to_utt[speaker_id]
            for j, utterance_file in enumerate(np.random.choice(utterances_files, size=num_per_speaker, replace=True)):
                self.load_into_mat(utterance_file, self.categorical_speakers, speaker_id, max_length, kx, ky,
                                   i * num_per_speaker + j)
        return kx, ky

    def generate(self, max_length=NUM_FRAMES, counts_per_speaker=(3000, 500)):
        kx_train, ky_train = self.generate_per_phase(max_length, counts_per_speaker[0], is_test=False)
        kx_test, ky_test = self.generate_per_phase(max_length, counts_per_speaker[1], is_test=True)
        logger.info(f'kx_train.shape = {kx_train.shape}')
        logger.info(f'ky_train.shape = {ky_train.shape}')
        logger.info(f'kx_test.shape = {kx_test.shape}')
        logger.info(f'ky_test.shape = {ky_test.shape}')
        self.kx_train, self.ky_train, self.kx_test, self.ky_test = kx_train, ky_train, kx_test, ky_test

    @staticmethod
    def load_into_mat(utterance_file, categorical_speakers, speaker_id, max_length, kx, ky, i):
        kx[i] = sample_from_mfcc_file(utterance_file, max_length)
        ky[i] = categorical_speakers.get_index(speaker_id)


class SparseCategoricalSpeakers:

    def __init__(self, speakers_list):
        self.speaker_ids = sorted(speakers_list)
        assert len(set(self.speaker_ids)) == len(self.speaker_ids)  # all unique.
        self.map = dict(zip(self.speaker_ids, range(len(self.speaker_ids))))

    def get_index(self, speaker_id):
        return self.map[speaker_id]

class LazyTripletBatcher:
    def __init__(self, working_dir: str, max_length: int, model: DeepSpeaker,device):
        self.working_dir = working_dir
        self.audio = Audio(cache_dir=working_dir)
        logger.info(f'Picking audio from {working_dir}.')
        self.sp_to_utt_train = train_test_sp_to_utt(self.audio, is_test=False)
        self.sp_to_utt_test = train_test_sp_to_utt(self.audio, is_test=True)
        self.max_length = max_length
        self.model = model.to(device)
        self.device=device
        self.nb_per_speaker = 2
        self.nb_speakers = 640
        self.history_length = 4
        self.history_every = 100  # batches.
        self.total_history_length = self.nb_speakers * self.nb_per_speaker * self.history_length  # 25,600
        self.metadata_train_speakers = Counter()
        self.metadata_output_file = os.path.join(self.working_dir, 'debug_batcher.json')

        self.history_embeddings_train = deque(maxlen=self.total_history_length)
        self.history_utterances_train = deque(maxlen=self.total_history_length)
        self.history_model_inputs_train = deque(maxlen=self.total_history_length)

        self.history_embeddings = None
        self.history_utterances = None
        self.history_model_inputs = None

        self.batch_count = 0
        for _ in tqdm(range(self.history_length), desc='Initializing the batcher'):  # init history.
            self.update_triplets_history()

    def update_triplets_history(self):
        model_inputs = []
        speakers = list(self.audio.speakers_to_utterances.keys())
        np.random.shuffle(speakers)
        selected_speakers = speakers[: self.nb_speakers]
        embeddings_utterances = []
        for speaker_id in selected_speakers:
            train_utterances = self.sp_to_utt_train[speaker_id]
            for selected_utterance in np.random.choice(a=train_utterances, size=self.nb_per_speaker, replace=False):
                mfcc = sample_from_mfcc_file(selected_utterance, self.max_length)
                embeddings_utterances.append(selected_utterance)
                model_inputs.append(mfcc)
        model_inputs=torch.tensor(model_inputs,device=self.device).float()
        model_inputs=torch.permute(model_inputs,(0,3,1,2))
#         print(model_inputs.shape)
        with torch.no_grad():
            embeddings = self.model(model_inputs)
            embeddings = embeddings.detach().cpu().numpy()
            
        assert embeddings.shape[-1] == 512
        embeddings = np.reshape(embeddings, (len(selected_speakers), self.nb_per_speaker, 512))
        self.history_embeddings_train.extend(list(embeddings.reshape((-1, 512))))
        self.history_utterances_train.extend(embeddings_utterances)
        self.history_model_inputs_train.extend(model_inputs.detach().cpu().numpy())

        # reason: can't index a deque with a np.array.
        self.history_embeddings = np.array(self.history_embeddings_train)
        self.history_utterances = np.array(self.history_utterances_train)
        self.history_model_inputs = np.array(self.history_model_inputs_train)

        with open(self.metadata_output_file, 'w') as w:
            json.dump(obj=dict(self.metadata_train_speakers), fp=w, indent=2)

    def get_batch(self, batch_size, is_test=False):
        return self.get_batch_test(batch_size) if is_test else self.get_random_batch(batch_size, is_test=False)

    def get_batch_test(self, batch_size):
        return self.get_random_batch(batch_size, is_test=True)

    def get_random_batch(self, batch_size, is_test=False):
        sp_to_utt = self.sp_to_utt_test if is_test else self.sp_to_utt_train
        speakers = list(self.audio.speakers_to_utterances.keys())
        anchor_speakers = np.random.choice(speakers, size=batch_size // 3, replace=False)

        anchor_utterances = []
        positive_utterances = []
        negative_utterances = []
        for anchor_speaker in anchor_speakers:
            negative_speaker = np.random.choice(list(set(speakers) - {anchor_speaker}), size=1)[0]
            assert negative_speaker != anchor_speaker
            pos_utterances = np.random.choice(sp_to_utt[anchor_speaker], 2, replace=False)
            neg_utterance = np.random.choice(sp_to_utt[negative_speaker], 1, replace=True)[0]
            anchor_utterances.append(pos_utterances[0])
            positive_utterances.append(pos_utterances[1])
            negative_utterances.append(neg_utterance)

        # anchor and positive should have difference utterances (but same speaker!).
        anc_pos = np.array([positive_utterances, anchor_utterances])
        assert np.all(anc_pos[0, :] != anc_pos[1, :])
        assert np.all(np.array([extract_speaker(s) for s in anc_pos[0, :]]) == np.array(
            [extract_speaker(s) for s in anc_pos[1, :]]))

        pos_neg = np.array([positive_utterances, negative_utterances])
        assert np.all(pos_neg[0, :] != pos_neg[1, :])
        assert np.all(np.array([extract_speaker(s) for s in pos_neg[0, :]]) != np.array(
            [extract_speaker(s) for s in pos_neg[1, :]]))

        batch_x = np.vstack([
            [sample_from_mfcc_file(u, self.max_length) for u in anchor_utterances],
            [sample_from_mfcc_file(u, self.max_length) for u in positive_utterances],
            [sample_from_mfcc_file(u, self.max_length) for u in negative_utterances]
        ])

        batch_y = np.zeros(shape=(len(batch_x), 1))  # dummy. sparse softmax needs something.
        batch_x=torch.from_numpy(batch_x).permute(0,3,1,2)
        batch_x.requires_grad_()
        batch_y=torch.from_numpy(batch_y)
        batch_y.requires_grad_()
        return batch_x, batch_y

    def get_batch_train(self, batch_size):
        # s1 = time()
        self.batch_count += 1
        if self.batch_count % self.history_every == 0:
            self.update_triplets_history()

        all_indexes = range(len(self.history_embeddings_train))
        anchor_indexes = np.random.choice(a=all_indexes, size=batch_size // 3, replace=False)

        # s2 = time()
        similar_negative_indexes = []
        dissimilar_positive_indexes = []
        # could be made parallel.
        for anchor_index in anchor_indexes:
            # s21 = time()
            anchor_embedding = self.history_embeddings[anchor_index]
            anchor_speaker = extract_speaker(self.history_utterances[anchor_index])

            # why self.nb_speakers // 2? just random. because it is fast. otherwise it's too much.
            negative_indexes = [j for (j, a) in enumerate(self.history_utterances)
                                if extract_speaker(a) != anchor_speaker]
            negative_indexes = np.random.choice(negative_indexes, size=self.nb_speakers // 2)

            # s22 = time()

            anchor_embedding_tile = [anchor_embedding] * len(negative_indexes)
            anchor_cos = batch_cosine_similarity(anchor_embedding_tile, self.history_embeddings[negative_indexes])

            # s23 = time()
            similar_negative_index = negative_indexes[np.argsort(anchor_cos)[-1]]  # [-1:]
            similar_negative_indexes.append(similar_negative_index)

            # s24 = time()
            positive_indexes = [j for (j, a) in enumerate(self.history_utterances) if
                                extract_speaker(a) == anchor_speaker and j != anchor_index]
            # s25 = time()
            anchor_embedding_tile = [anchor_embedding] * len(positive_indexes)
            # s26 = time()
            anchor_cos = batch_cosine_similarity(anchor_embedding_tile, self.history_embeddings[positive_indexes])
            dissimilar_positive_index = positive_indexes[np.argsort(anchor_cos)[0]]  # [:1]
            dissimilar_positive_indexes.append(dissimilar_positive_index)
            # s27 = time()

        # s3 = time()
        batch_x = np.vstack([
            self.history_model_inputs[anchor_indexes],
            self.history_model_inputs[dissimilar_positive_indexes],
            self.history_model_inputs[similar_negative_indexes]
        ])

        
        anchor_speakers = [extract_speaker(a) for a in self.history_utterances[anchor_indexes]]
        positive_speakers = [extract_speaker(a) for a in self.history_utterances[dissimilar_positive_indexes]]
        negative_speakers = [extract_speaker(a) for a in self.history_utterances[similar_negative_indexes]]

        assert len(anchor_indexes) == len(dissimilar_positive_indexes)
        assert len(similar_negative_indexes) == len(dissimilar_positive_indexes)
        assert list(self.history_utterances[dissimilar_positive_indexes]) != list(
            self.history_utterances[anchor_indexes])
        assert anchor_speakers == positive_speakers
        assert negative_speakers != anchor_speakers

        batch_y = np.zeros(shape=(len(batch_x), 1))  # dummy. sparse softmax needs something.

        for a in anchor_speakers:
            self.metadata_train_speakers[a] += 1
        for a in positive_speakers:
            self.metadata_train_speakers[a] += 1
        for a in negative_speakers:
            self.metadata_train_speakers[a] += 1
        batch_x=torch.from_numpy(batch_x).permute(0,3,1,2)
        batch_x.requires_grad_()
        batch_y=torch.from_numpy(batch_y)
        batch_y.requires_grad_()
        return batch_x, batch_y

    def get_speaker_verification_data(self, anchor_speaker, num_different_speakers):
        speakers = list(self.audio.speakers_to_utterances.keys())
        anchor_utterances = []
        positive_utterances = []
        negative_utterances = []
        negative_speakers = np.random.choice(list(set(speakers) - {anchor_speaker}), size=num_different_speakers)
        assert [negative_speaker != anchor_speaker for negative_speaker in negative_speakers]
        pos_utterances = np.random.choice(self.sp_to_utt_test[anchor_speaker], 2, replace=False)
        neg_utterances = [np.random.choice(self.sp_to_utt_test[neg], 1, replace=True)[0] for neg in negative_speakers]
        anchor_utterances.append(pos_utterances[0])
        positive_utterances.append(pos_utterances[1])
        negative_utterances.extend(neg_utterances)

        # anchor and positive should have difference utterances (but same speaker!).
        anc_pos = np.array([positive_utterances, anchor_utterances])
        assert np.all(anc_pos[0, :] != anc_pos[1, :])
        assert np.all(np.array([extract_speaker(s) for s in anc_pos[0, :]]) == np.array(
            [extract_speaker(s) for s in anc_pos[1, :]]))

        batch_x = np.vstack([
            [sample_from_mfcc_file(u, self.max_length) for u in anchor_utterances],
            [sample_from_mfcc_file(u, self.max_length) for u in positive_utterances],
            [sample_from_mfcc_file(u, self.max_length) for u in negative_utterances]
        ])

        batch_y = np.zeros(shape=(len(batch_x), 1))  # dummy. sparse softmax needs something.
        batch_x=torch.from_numpy(batch_x).permute(0,3,1,2)
        batch_x.requires_grad_()
        batch_y=torch.from_numpy(batch_y)
        batch_y.requires_grad_()
        return batch_x, batch_y

class DeepSpeakerDatasetSoftmax(Dataset):
  def __init__(self,kx,ky):
    self.kx=np.load(kx)
    self.ky=np.load(ky)
  def __len__(self):
    return len(self.kx)

  def __getitem__(self,idx):
    x=torch.from_numpy(self.kx[idx])
    x=torch.permute(x,(2,0,1))
    y=torch.from_numpy(self.ky[idx]).long()
    return x,y