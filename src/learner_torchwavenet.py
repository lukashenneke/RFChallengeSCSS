# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import h5py

from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from .config_torchwavenet import Config
from .dataset import MixtureDataset
from .torchwavenet import Wave
from .soi_autoencoder import WaveQPSK, WaveOFDM
from .loss import MSE_BCE_loss, MSE_MSE_loss

_map = lambda x: torch.view_as_real(x).transpose(-2,-1)

class WaveLearner:
    def __init__(self, cfg: Config, model: nn.Module, gen):
        self.cfg = cfg
        self.gen = gen

        # Store some import variables
        self.model_dir = cfg.model_dir
        self.log_every = cfg.trainer.log_every
        self.validate_every = cfg.trainer.validate_every
        self.save_every = cfg.trainer.save_every
        self.max_steps = cfg.trainer.max_steps
        self.build_dataloaders()

        self.model = model
        params = [self.model] if not cfg.trainer.fix_wavenet_params else [self.model.encoder, self.model.scale, self.model.decoder]
        self.optimizer = torch.optim.Adam(
            (p for m in params for p in m.parameters()), lr=cfg.trainer.learning_rate)
        if len(cfg.trainer.lr_milestones) == 0:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min",)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, cfg.trainer.lr_milestones, gamma=0.1, last_epoch=-1)
        self.autocast = torch.cuda.amp.autocast(enabled=cfg.trainer.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer.fp16)
        self.step = 0

        if cfg.trainer.loss_fun == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif cfg.trainer.loss_fun == 'MSE+BCE':
            self.loss_fn = MSE_BCE_loss(cfg.trainer.lambda_mse, cfg.trainer.lambda_ber)
        elif cfg.trainer.loss_fun == 'MSE+MSE':
            self.loss_fn = MSE_MSE_loss(cfg.trainer.lambda_mse, cfg.trainer.lambda_ber)
        self.writer = SummaryWriter(self.model_dir)

    def build_dataloaders(self):
        bits = not (self.cfg.trainer.loss_fun == 'MSE')
        cdt = self.cfg.data
        with h5py.File(self.cfg.data.data_dir,'r') as data_h5file:
            sig_data = np.array(data_h5file.get('dataset'))
        with h5py.File(self.cfg.data.val_data_dir,'r') as data_h5file:
            valid_data = np.array(data_h5file.get('dataset'))
        self.train_dataset = MixtureDataset(
            self.gen, sig_data, cdt.sig_len, cdt.batch_size*1000, cdt.sinr_range, cdt.fo_std, return_bits=bits, fix=False
        )
        self.val_dataset = MixtureDataset(
            self.gen, valid_data, cdt.sig_len, cdt.batch_size*64, cdt.sinr_range, cdt.fo_std, return_bits=bits, fix=True
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cdt.batch_size,
            shuffle=False,
            num_workers=cdt.num_workers,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=cdt.batch_size * 4,
            shuffle=False,
            num_workers=cdt.num_workers,
            pin_memory=True,
        )

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) 
                      else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) 
                          else v for k, v in self.optimizer.state_dict().items()},
            'cfg': asdict(self.cfg),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']
        if len(self.cfg.trainer.lr_milestones) > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.cfg.trainer.lr_milestones, gamma=0.1, last_epoch=self.step-1
            )

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self):

        while True:
            for features in tqdm(
                self.train_dataloader, 
                desc=f"Training ({self.step} / {self.max_steps})"
            ):
                if self.step >= self.max_steps:
                    self.save_to_checkpoint()
                    print("Ending training...")
                    exit(0)

                loss = self.train_step(features)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at step {self.step}.')

                if self.step % self.log_every == 0:
                    self.writer.add_scalar('train/loss', loss, self.step)
                    if hasattr(self.loss_fn, 'tmp_ber'):
                        self.writer.add_scalar('train/loss_mse', self.loss_fn.tmp_mse, self.step)
                        self.writer.add_scalar('train/loss_ber', self.loss_fn.tmp_ber, self.step)
                    self.writer.add_scalar('train/grad_norm', self.grad_norm, self.step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]["lr"], self.step)
                if self.step % self.save_every == 0:
                    self.save_to_checkpoint()

                if self.step % self.validate_every == 0:
                    val_loss = self.validate()
                    # Update the learning rate if it plateus
                    if len(self.cfg.trainer.lr_milestones) == 0:
                        self.lr_scheduler.step(val_loss)

                if len(self.cfg.trainer.lr_milestones) > 0:
                    self.lr_scheduler.step()

                self.step += 1

    def train_step(self, features: tuple):
        device = next(self.model.parameters()).device
        for param in self.model.parameters():
            param.grad = None

        sample_mix = _map(features[0]).to(device)
        target = _map(features[1]).to(device)
        if len(features) > 2: target = (target, features[2].to(device))

        with self.autocast:
            predicted = self.model(sample_mix)
            loss = self.loss_fn(predicted, target)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.trainer.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        loss = 0
        for features in tqdm(
            self.val_dataloader, 
            desc=f"Running validation after step {self.step}"
        ):
            sample_mix = _map(features[0]).to(device)
            target = _map(features[1]).to(device)
            if len(features) > 2: target = (target, features[2].to(device))

            with self.autocast:
                predicted = self.model(sample_mix)
                loss +=  self.loss_fn(predicted, target) * sample_mix.shape[0] / len(self.val_dataset)

        self.writer.add_scalar('val/loss', loss, self.step)
        self.model.train()

        return loss
    

def train(cfg: Config, gen):
    """Training on a single GPU."""
    torch.backends.cudnn.benchmark = True

    model = Wave(cfg.model).cuda()
    print('Params:', sum(p.numel() for p in model.parameters()))

    learner = WaveLearner(cfg, model, gen)
    learner.restore_from_checkpoint()
    learner.train()

def train_sae(cfg: Config, gen):
    torch.backends.cudnn.benchmark = True

    if gen.soi_type == 'QPSK':
        model = WaveQPSK(cfg.model, cfg.sae).cuda()
    elif gen.soi_type == 'OFDMQPSK':
        model = WaveOFDM(cfg.model, cfg.sae).cuda()
    if cfg.pretrained is not None:
        res = model.load_state_dict(torch.load(cfg.pretrained)['model'], strict=False)
    print('Params:', sum(p.numel() for p in model.parameters()))

    learner = WaveLearner(cfg, model, gen)
    learner.restore_from_checkpoint()
    learner.train()
