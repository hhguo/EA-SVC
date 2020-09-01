from tqdm import tqdm

import numpy as np
import os
import random
import torch
import torch.nn.functional as F

import utils.utils as utils


class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, file_list, ppg_dir, f0_dir, audio_dir, se_files,
                 pitch_norm, hop_length, segment_length, sampling_rate):
        self.pitch_norm = pitch_norm
        self.segment_length = segment_length
        self.segment_n_frames = segment_length // hop_length
        self.sampling_rate = sampling_rate
        random.seed(1234)

        file_list = utils.files_to_list(file_list)
        random.shuffle(file_list)
        self.file_list = file_list

        print("Loading audios...")
        audio_files = [os.path.join(audio_dir, x + '.wav') for x in file_list]
        audio_files = [utils.load_wav_to_torch(x)[0] for x in tqdm(audio_files)]
        self.audio_files = audio_files
       
        print("Loading phonetic posteriorgrams...")
        ppg_files = [os.path.join(ppg_dir, x + '.npy') for x in file_list]
        ppg_files = [self.parse_ppg_file(x) for x in tqdm(ppg_files)]
        self.ppg_files = ppg_files
        
        print("Loading pitch (F0)...")
        f0_files = [os.path.join(f0_dir, x + '.f0') for x in file_list]
        f0_files = [self.parse_f0_file(x) for x in tqdm(f0_files)]
        self.f0_files = f0_files
            
        print("Loading speaker embedding...")
        se_files = self.parse_se_file(se_files, file_list)
        self.se_files = se_files

    def parse_ppg_file(self, ppg_file):
        ppg = np.load(ppg_file).astype(np.float32)
        return ppg

    def parse_f0_file(self, f0_file):
        with open(f0_file) as fin:
            lines = fin.readlines()
        f0 = np.asarray([float(x.strip()) for x in lines]) / self.pitch_norm
        return np.expand_dims(np.asarray(f0), -1).astype(np.float32)
    
    def parse_se_file(self, se_file, train_list):
        se_dict = {}
        with open(se_file) as fin:
            for line in fin.readlines():
                segs = line.strip().split()
                se_dict[segs[0]] = np.asarray([float(x)
                    for x in segs[2: -1]]).astype(np.float32)
        return np.asarray([se_dict[x] for x in train_list])

    def parse_input(self, index):
        cond = []
       
        cond.append(self.ppg_files[index])
        cond.append(self.f0_files[index])

        se = self.se_files[index]
        se = np.repeat(np.expand_dims(se, 0),
                       self.f0_files[index].shape[0], 0)
        cond.append(se)

        n_frames = min([x.shape[0] for x in cond])
        return np.concatenate([x[: n_frames] for x in cond], axis=-1)
    
    def __getitem__(self, index):
        # Read data
        audio = self.audio_files[index]
        cond = self.parse_input(index)
        # Take segment
        max_start = cond.shape[0] * self.hop_length - (
                    self.segment_length + self.hop_length * 10)
        if max_start >= 0:
            frame_start = random.randint(0, max_start) // self.hop_length
            sample_start = frame_start * self.hop_length
            cond = cond[frame_start: frame_start + self.segment_n_frames]
            cond = np.transpose(cond, (1, 0))
            audio = audio[sample_start: sample_start + self.segment_length]
        else:
            print("The audio file is too short! {}".format(max_start))
            raise RuntimeError
        return (cond, audio)
    
    def __len__(self):
        return len(self.file_list)
