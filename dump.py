import torch
import numpy as np
import os
from subprocess import call
import pandas as pd
import pdb
import librosa
import torchaudio
from tqdm import tqdm

class UttStore(object):
    def __init__(self):
        self.audio_file = ''
        self.speaker_map = {'A':0, 'B':1}

    def update(self, audio_file):
        if audio_file != self.audio_file:
            self.audio_file = audio_file
            self.signal, self.sr = self._load(audio_file)

    def _load(self, path):
        if path.split('.')[-1] != 'wav':
            call(f'./sph2pipe {path} temp.wav', shell=True)
            signal, sr = torchaudio.load('temp.wav')
            os.remove('temp.wav')
        else:
            signal, sr = torchaudio.load(path)
        return signal, sr

    def _relevant(self, signal, sr, start, end):
        crop_start_ix = librosa.core.time_to_samples(start, sr=sr)
        crop_end_ix = librosa.core.time_to_samples(end, sr=sr)
        return signal[crop_start_ix:crop_end_ix]

    def _signal(self, start, end, speaker='A'):
        if len(self.signal.size()) > 1:
            signal = self.signal[self.speaker_map[speaker]]
        if start != 0.0 or end != 0.0:
            signal = self._relevant(signal, self.sr, start, end)
        return signal.unsqueeze(0)

    def save_rep(self, cnum, utt_id, start, end, spath, speaker='A'):
        signal = self._signal(start, end, speaker=speaker)
        with open(os.path.join(spath,f'{cnum}_{utt_id}.npz'), 'wb') as f:
            np.savez_compressed(f, a=signal)

def get_path_hvb(ab,rel):
    x,y = ab.split('_')
    return os.path.join(rel,x,y)


path_csv = '/data/data25/scratch/sunderv/fisher/fisher_full.csv' #path to csv
path_save = '/data/data24/scratch/sunderv/fisher/' #path where features should be saved
df = pd.read_csv(path_csv)
store = UttStore()
for i, row in tqdm(df.iterrows(),total=df.shape[0]):
    store.update(row['audio_file'])
    #store.update(get_path_hvb(row['audio_file'], '/homes/3/sunder.9/hvb/data/audio/audio/'))
    store.save_rep(str(row['cnum']), str(row['utt_id']), row['begin'], row['end'], path_save, row['speaker'])
