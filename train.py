from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import init_distributed, apply_gradient_allreduce
from models.disentangler import Disentangler
from models.encoder import Encoder
from models.generator import Generator
from models.multiscale import MultiScaleDiscriminator
from utils.dataset import Dataset
from utils.loss import MultiResolutionSTFTLoss
from utils.optimizer import *
from utils.utils import to_gpu, LossMeter


def clean_checkpoint_directory(checkpoint_path):
    checkpoint_dir, model_name = os.path.split(checkpoint_path)
    prefix, iterations = model_name.split('_')
    iterations = int(iterations)
    for filename in os.listdir(checkpoint_dir):
        if prefix not in filename:
            continue
        iters = int(filename.split('_')[-1])
        if iters % 50000 != 0 and iterations - iters > 10000:
            os.remove(os.path.join(checkpoint_dir, filename))


def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    try:
        g_optimizer.load_state_dict(checkpoint_dict['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint_dict['d_optimizer'])
        model.load_state_dict(checkpoint_dict['model'])
    except:
        print('Loaded model is not the same as the current one')
        g_optimizer.load_state_dict(checkpoint_dict['g_optimizer'])
        model.load_state_dict(checkpoint_dict['model'], strict=False)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, g_optimizer, d_optimizer, iteration


def save_checkpoint(model, g_optimizer, d_optimizer, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    clean_checkpoint_directory(filepath)
    torch.save({'model': model.state_dict(),
                'iteration': iteration,
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict()}, filepath)

    
def train(num_gpus, rank, group_name, output_directory, epochs,
          g_learning_rate, d_learning_rate, adv_ag, adv_fd,
          lamda_adv, lamda_feat, warmup_steps, decay_learning_rate,
          iters_per_checkpoint, batch_size, seed, checkpoint_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    model = torch.nn.Module()
    model.add_module('encoder', Encoder(**encoder_config))
    model.add_module('generator', Generator(sum(encoder_config['n_out_channels'])))
    model.add_module('discriminator', MultiScaleDiscriminator(**discriminator_config))
    model.add_module('disentangler', Disentangler(encoder_config['n_out_channels'][0],
                                                  sum(encoder_config['n_out_channels'][1:])))
    model = model.cuda()

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    g_parameters = list(model.generator.parameters())
    g_parameters = list(model.encoder.parameters()) + g_parameters
    g_optimizer = Lookahead(RAdam(g_parameters, lr=g_learning_rate))

    d_parameters = list(model.discriminator.parameters())
    d_parameters = list(model.disentangler.parameters()) + d_parameters
    d_optimizer = Lookahead(RAdam(d_parameters, lr=d_learning_rate))
    
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, g_optimizer, d_optimizer, iteration = load_checkpoint(
                checkpoint_path, model, g_optimizer, d_optimizer)
        iteration += 1  # next iteration is iteration + 1
    
    customer_g_optimizer = Optimizer(g_optimizer, g_learning_rate,
                iteration, warmup_steps, decay_learning_rate)
    customer_d_optimizer = Optimizer(d_optimizer, d_learning_rate,
                iteration, warmup_steps, decay_learning_rate)

    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()
    stft_criterion = MultiResolutionSTFTLoss()

    trainset = Dataset(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
        logdir = os.path.join(output_directory,
            time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir=logdir)
        anchors = [
            'loss_g', 'loss_g_sc', 'loss_g_mag', 'loss_g_adv', 'loss_g_feat',
            'loss_g_fd', 'loss_d', 'loss_d_real', 'loss_d_fake', 'loss_d_fd']
        meters = {x: LossMeter(x, writer, 100, iteration, True)
                  for x in anchors}
    
    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        train_sampler.set_epoch(epoch) if train_sampler is not None else None
        tbar = tqdm(enumerate(train_loader)) if rank == 0 else enumerate(train_loader)
        for i, batch in tbar:
            model.zero_grad()
            
            cond, s, a = [to_gpu(x) for x in batch]
           
            # Get generator outputs
            x = model.encoder(cond)
            g_outputs = model.generator(x)
            
            losses = {}

            # Get Discrimiantor loss
            customer_d_optimizer.zero_grad()
            d_loss = []
            # Adversarial training for audio generation
            if adv_ag == True:
                real_scores, _ = model.discriminator(a.unsqueeze(1))
                fake_scores, _ = model.discriminator(g_outputs.detach())

                d_loss_fake_list, d_loss_real_list = [], []
                for (real_score, fake_score) in zip(real_scores, fake_scores):
                    d_loss_real_list.append(criterion(real_score, torch.ones_like(real_score)))
                    d_loss_fake_list.append(criterion(fake_score, torch.zeros_like(fake_score)))

                d_loss_real = sum(d_loss_real_list) / len(d_loss_real_list)
                d_loss_fake = sum(d_loss_fake_list) / len(d_loss_fake_list)
                d_loss = d_loss + [d_loss_real, d_loss_fake]
                losses.update({'loss_d_real': d_loss_real,
                               'loss_d_fake': d_loss_fake})
            # Adversarial training for feature disentanglement
            if adv_fd == True:
                split_x = torch.split(x.detach(), encoder_config['n_out_channels'], dim=1)
                pred = model.disentangler(split_x[0])
                d_loss_fd = F.l1_loss(pred, torch.cat((split_x[1: ]), dim=1))
                d_loss = d_loss + [d_loss_fd]
                losses.update({'loss_d_fd': d_loss_fd})
            if len(d_loss) > 0:
                d_loss = sum(d_loss)
                d_loss.backward()
                nn.utils.clip_grad_norm_(d_parameters, max_norm=10)
                customer_d_optimizer.step_and_update_lr()
                losses.update({'loss_d': d_loss})

            # Get generator loss
            customer_g_optimizer.zero_grad()
            g_clip_norm_scale = 10
            # STFT Loss
            sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), a)
            g_loss = sc_loss + mag_loss
            losses.update({'loss_g_sc': sc_loss, 'loss_g_mag': mag_loss})
            # Adversarial training for audio generation
            if adv_ag == True:
                fake_scores, fake_feats = model.discriminator(g_outputs)
                real_scores, real_feats = model.discriminator(a.unsqueeze(1))
                
                adv_loss_list, feat_loss_list = [], []
                for i, fake_score in enumerate(fake_scores):
                    adv_loss_list.append(
                        criterion(fake_score, torch.ones_like(fake_score)))
                adv_loss = sum(adv_loss_list) / len(adv_loss_list)
                
                for i in range(len(fake_feats)):
                    for j in range(len(fake_feats[i])):
                        feat_loss_list.append(l1_loss(
                            fake_feats[i][j], real_feats[i][j].detach()))
                feat_loss = sum(feat_loss_list) / len(feat_loss_list)
                
                g_loss = g_loss + adv_loss * lamda_adv + feat_loss * lamda_feat
                losses.update({'loss_g_adv': adv_loss})
                losses.update({'loss_g_feat': feat_loss})
                g_clip_norm_scale = 0.5
            # Adversarial training for feature disentanglement
            if adv_fd == True:
                split_x = torch.split(x, encoder_config['n_out_channels'], dim=1)
                pred = model.disentangler(split_x[0])
                g_loss_fd = F.l1_loss(
                        pred, torch.cat((split_x[1: ]), dim=1).detach())
                g_loss = g_loss + (-1.0) * g_loss_fd
                losses.update({'loss_g_fd': g_loss_fd})
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=g_clip_norm_scale)
            customer_g_optimizer.step_and_update_lr()
            losses.update({'loss_g': g_loss})

            # only output log of 0-th GPU
            if rank == 0:
                tbar.set_description("{:>7}:  ".format(iteration) + ', '.join(
                    ["{}: {:.1e}".format(x[5:], losses[x].item())
                     for x in losses.keys()]))
                for x in losses:
                    meters[x].add(losses[x].item())
                if (iteration % iters_per_checkpoint == 0):
                    checkpoint_path = "{}/model_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, g_optimizer, d_optimizer,
                                    iteration, checkpoint_path)
                     
            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()
    
    # Parse configs.
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
   
    # Parse global configs.
    global data_config, dist_config, encoder_config, discriminator_config
    data_config = config["data_config"]
    dist_config = config["dist_config"]
    encoder_config = config["encoder_config"]
    discriminator_config = config["discriminator_config"]

    # Single GPU or Multi GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    # Begin Training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
