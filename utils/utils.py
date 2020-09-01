from scipy.io.wavfile import read

import os
import torch
import numpy as np


MAX_WAV_VALUE = 32768.0


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)) / MAX_WAV_VALUE, sampling_rate


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()
    
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


class LossMeter():

    def __init__(self, name, writer, log_per_step, init_step=0, auto_log=True):
        self.name = name
        self.writer = writer
        self.log_per_step = log_per_step
        self.step = init_step
        self.auto_log = auto_log
        self.loss = []

    def add(self, loss):
        self.step += 1
        assert isinstance(loss, float), 'Loss must be float type'
        self.loss.append(loss)

        if self.auto_log and self.step % self.log_per_step == 0:
            self.writer.add_scalar(self.name, self.mean(), self.step)
            self.reset()

    def reset(self):
        self.loss = []

    def mean(self):
        return self.sum() / len(self.loss)

    def sum(self):
        return sum(self.loss)
