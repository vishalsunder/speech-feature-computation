import torch
import os
import time
import pandas as pd
import pdb
import numpy as np
import torchaudio
from tqdm import tqdm
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        self.df = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.compute_deltas = Deltas(input_size=n_mels)
        self.compute_cw = ContextWindow(left_frames=0, right_frames=1)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        key = str(row['cnum'])+'_'+str(row['utt_id'])
        audio_path = os.path.join(self.audio_path,f'{key}.npz')
        signal = torch.from_numpy(np.load(audio_path)['a'])
        #signal = torch.from_numpy(self.data["audio"][self.id2key[index]][:])
        #signal = torch.from_numpy(self.data[self.id2key[index]][:])
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        delta1 = self.compute_deltas(features)
        delta2 = self.compute_deltas(delta1)
        features = torch.cat([features, delta1, delta2], dim=2)
        features  = self.compute_cw(features).squeeze()
        return features[::2]

def collate_speech(speech_list):
    return speech_list

csv_path = '/data/data25/scratch/sunderv/fisher/fisher_full.csv'#'/data/data25/scratch/sunderv/fisher/fisher_full.csv'#'/data/data25/scratch/sunderv/hvb/conv_train.csv'#'/data/data24/scratch/sunderv/hvb/train_raw.hdf5'
audio_path = '/data/data24/scratch/sunderv/fisher/'
data = SpeechDataset(csv_path, audio_path)
generator = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_speech)

for x in tqdm(generator):
    time.sleep(1)
