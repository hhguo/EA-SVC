from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import write
from tqdm import tqdm

import argparse
import json
import numpy as np
import os
import re
import torch

import utils.utils as utils

from models.encoder import Encoder
from models.generator import Generator
from models.multiscale import MultiScaleDiscriminator
from utils.dataset import Dataset


use_predicted_pitch = False


def chunker(testset, size):
    """
    https://stackoverflow.com/a/434328
    """
    seq = [testset[i] for i in range(len(testset))]
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    match = re.compile(r'.*_([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(match.group(1)) if match else 'eval'
    return os.path.join(base_dir, name)


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint_dict['model'], strict=True)
    except:
        model.load_state_dict(checkpoint_dict['model'], strict=False)
    print("Loaded checkpoint '{}'" .format(checkpoint_path))
    return model


def adapt_f0(s, t):
    if use_predicted_pitch:
        s = utils.to_gpu(torch.from_numpy(s)).view(1, -1, 1).float()
        t = utils.to_gpu(torch.from_numpy(t)).view(1, -1, 1).float()
        s = pitch_model(s, t)[0, :].cpu().numpy()
        return s
    else:
        tmp_s = np.asarray([x for x in s if x > 0]).mean()
        tmp_t = np.asarray([x for x in t if x > 0]).mean()
        for i in range(s.shape[0]):
            if s[i] > 0:
                s[i] = s[i] * tmp_t / tmp_s
        return s


class TestSet(Dataset):

    def __init__(self, file_list, ppg_dir, f0_dir, audio_dir, sp_dir, se_files,
                 feat_used, pitch_norm, segment_length, mu_quantization,
                 filter_length, hop_length, win_length, sampling_rate):
        self.feat_used = feat_used
        self.pitch_norm = pitch_norm
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_n_frames = segment_length // hop_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate
        
        data_dir = os.path.dirname(file_list)
        sp_dir = os.path.join(data_dir, 'mels')
        f0_dir = os.path.join(data_dir, 'f0_reaper')
        se_files = os.path.join(data_dir, 'utt_emb_sing3.ark')
        
        file_list = utils.files_to_list(file_list)
        self.file_list = ['_'.join(x.split('|')) for x in file_list]
        ppg_list, f0_list, se_list = zip(*[x.split('|') for x in file_list])

        if 'p' in feat_used:
            ppg_files = [os.path.join(ppg_dir, x + '.npy') for x in ppg_list]
            ppg_files = [self.parse_ppg_file(x) for x in tqdm(ppg_files)]
            self.ppg_files = ppg_files
        
        if 'f' in feat_used:
            f0_files = [os.path.join(f0_dir, x + '.f0') for x in ppg_list]
            f0_files = [self.parse_f0_file(x) for x in tqdm(f0_files)]
            
            target_f0 = [os.path.join(f0_dir, x + '.f0') for x in se_list]
            target_f0 = [self.parse_f0_file(x) for x in tqdm(target_f0)]

            f0_files = [adapt_f0(x, target_f0[i]) for i, x in enumerate(f0_files)]
            
            f0_files = [x * float(f0_list[i]) for i, x in enumerate(f0_files)]
            self.f0_files = f0_files
        
        if 's' in feat_used:
            se_files = self.parse_se_file(se_files, se_list)
            self.se_files = se_files
        
        if 'a' in feat_used and encoder_config['speaker_input'] == 'audio':
            ref_files = [os.path.join(sp_dir, x + '.npy') for x in tqdm(se_list)]
            self.ref_files =[np.load(x) for x in ref_files]

    def __getitem__(self, index):
        cond = self.parse_input(index).transpose(1, 0)
        name = self.file_list[index]
        if hasattr(self, 'ref_files'):
            return cond, self.ref_files[index], name
        return cond, name
    
    def parse_se_file(self, se_file, train_list):
        se_dict = {}
        with open(se_file) as fin:
            for line in fin.readlines():
                segs = line.strip().split()
                se_dict[segs[0]] = np.asarray([float(x) for x in segs[2: -1]])
        outputs = []
        for x in train_list:
            if x not in se_dict:
                x = '_'.join(x.split('_')[: -1])
            outputs.append(se_dict[x])
        return np.asarray(outputs)


def main(model_filename, pitch_model_filename, output_dir, batch_size):
    model = torch.nn.Module()
    model.add_module('encoder', Encoder(**encoder_config))
    model.add_module('generator', Generator(sum(encoder_config['n_out_channels'])))
    model = load_checkpoint(model_filename, model).cuda()
    model.eval()
    
    if os.path.isfile(pitch_model_filename):
        global pitch_model, use_predicted_pitch
        use_predicted_pitch = True
        pitch_model = PitchModel(**pitch_config)
        pitch_model = load_checkpoint(pitch_model_filename, pitch_model).cuda()
        pitch_model.eval()

    testset = TestSet(**(data_config))
    for files in chunker(testset, batch_size):
        files = list(zip(*files))
        cond_input, file_paths = files[: -1], files[-1]
        cond_input = [utils.to_gpu(torch.from_numpy(np.stack(x))).float()
                      for x in cond_input]

        #cond_input = model.encoder(cond_input.transpose(1, 2)).transpose(1, 2)
        cond_input = model.encoder(cond_input[0])
        audio = model.generator(cond_input)

        for i, file_path in enumerate(file_paths):
            print("writing {}".format(file_path))
            wav = audio[i].cpu().squeeze().detach().numpy() * 32768.0
            write("{}/{}.wav".format(output_dir, file_path),
                  data_config['sampling_rate'], wav.astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", required=True)
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-m', "--checkpoint_path", required=True)
    parser.add_argument('-p', "--pitch_checkpoint_path", default='')
    parser.add_argument('-o', "--output_dir", default='')
    parser.add_argument('-b', "--batch_size", default=1)
    args = parser.parse_args()

    if args.output_dir == '':
        args.output_dir = get_output_base_path(args.checkpoint_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    global data_config, encoder_config, decoder_config, postnet_config
    data_config = config["data_config"]
    data_config['file_list'] = args.filelist_path
    encoder_config = config["encoder_config"]
    if "pitch_config" in config:
        global pitch_config
        pitch_config = config["pitch_config"]
    with torch.no_grad(): 
        main(args.checkpoint_path, args.pitch_checkpoint_path, args.output_dir, args.batch_size)
