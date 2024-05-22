import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
import wandb
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        proxy_model,
        dataset,
        proxy_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        proxy_train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        proxy_log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        proxy_save_freq=100,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        load_checkpoint=None,
        load_proxy_checkpoint=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.proxy_model = proxy_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.proxy_log_freq = proxy_log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.proxy_save_freq = proxy_save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.proxy_dataset = proxy_dataset
        
        ranks = torch.argsort(torch.argsort(-1 * self.proxy_dataset.data_y.flatten()))
        weights = 1.0 / (1e-2 * len(self.proxy_dataset.data_y) + ranks)
        sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(self.proxy_dataset.data_y), replacement=True
                )

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        ))
        self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
            self.proxy_dataset, batch_size=train_batch_size, num_workers=0, sampler=sampler, pin_memory=True, drop_last=True,
        ))
        # self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.proxy_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        # ))
        
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.proxy_optimizer = torch.optim.Adam(proxy_model.parameters(), lr=proxy_train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0
        self.proxy_step = 0

        self.device = train_device
        
        if load_checkpoint is not None:
            self.load(epoch=load_checkpoint)
            self.step = load_checkpoint
            
        if load_proxy_checkpoint is not None:
            self.proxy_load(epoch=load_proxy_checkpoint)
            self.proxy_step = load_proxy_checkpoint

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    
    def train_proxy(self, n_train_steps):
        timer = Timer()
        for step in tqdm(range(n_train_steps)):
            x, y = next(self.proxy_dataloader)
            # print(x[:4, :10])
            # print(y[:4])
            # print(self.proxy_dataset.normalizer.unnormalize(x[:4]))
            # print(self.proxy_dataset.normalizer_values.unnormalize(y[:4]))
            # print(kyle)
            # print(batch[0].shape, batch[1].shape)
            # batch = batch_to_device(batch, device=self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            loss, infos = self.proxy_model.loss(x, y)
            loss.backward()
            
            self.proxy_optimizer.step()
            self.proxy_optimizer.zero_grad()
            
            if self.proxy_step % self.proxy_log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.proxy_step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['proxy_steps'] = self.proxy_step
                metrics['proxy_loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')
                wandb.log(metrics)

            self.proxy_step += 1

            if self.proxy_step % self.proxy_save_freq == 0:
                self.proxy_save()

    def train(self, n_train_steps):

        timer = Timer()
        # for step in tqdm(range(n_train_steps)):
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')
                wandb.log(metrics)

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
            #         self.inv_render_samples()
            #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
            #         pass
            #     else:
            #         self.render_samples()

            self.step += 1
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()
                
    def sample(self, n_samples, topk, horizon, context_length, inpainting, guidance, confidence=False):
        new_trajectories = []
        new_trajectories_confidence_score = []
        for step in tqdm(range(0, n_samples, self.batch_size)):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            
            conditions = {i: to_torch(batch.trajectories[:, i], device=self.device) for i in range(context_length)}
            conditions["ctx_len"] = to_torch(np.ones(self.batch_size,), device=self.device) * context_length
           
            if inpainting:
                values = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                values = values.repeat(self.batch_size, horizon-context_length)
                # values = torch.linspace(batch.trajectories[Config.horizon-context_length-1, -1], 1.0, steps=Config.horizon).to(device=device).unsqueeze(0)
            else:
                values = None
            
            if guidance:
                returns = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                returns = returns.repeat(self.batch_size, 1)
                # returns = (to_torch(batch.trajectories[:, :context_length, -1].sum(axis=-1, keepdims=True), device=self.device) + horizon - context_length) / horizon
            else:
                # returns = torch.ones(1, ).to(device=self.device).unsqueeze(0) * 0.0
                # returns = returns.repeat(self.batch_size, 1)
                returns = None

            new_trajectory = self.ema_model.conditional_sample(conditions, values=values, returns=returns)
            # new_trajectory = self.ema_model.back_and_forth_sample(batch.trajectories, conditions, values=values, returns=returns)
            new_observation = self.proxy_dataset.normalizer.normalize(self.dataset.normalizer.unnormalize(new_trajectory[..., context_length:, :-1].cpu()).reshape(self.batch_size*(horizon - context_length), -1)).to(self.device)
            new_trajectory_score, new_trajectory_confidence_score = self.proxy_model(new_observation, confidence=True)
            # print(new_trajectory_score.flatten()[:10])
            new_trajectory_score = self.dataset.normalizer_values.normalize(self.proxy_dataset.unnormalize_values(new_trajectory_score.cpu())).to(self.device)
            # print(new_trajectory[:, context_length:, -1].flatten()[:10])
            # print(new_trajectory_score.flatten()[:10])
            # print(kyle)
            new_trajectory[..., context_length:, -1] = new_trajectory_score.reshape(self.batch_size, horizon-context_length)
            new_trajectory_confidence_score = new_trajectory_confidence_score.reshape(self.batch_size, horizon-context_length)
            
            new_trajectory = new_trajectory.cpu().detach()
            new_trajectory_confidence_score = new_trajectory_confidence_score.cpu().detach()
            
            new_trajectories.append(new_trajectory)
            new_trajectories_confidence_score.append(new_trajectory_confidence_score)
        new_trajectories = torch.cat(new_trajectories, dim=0)
        # print(new_trajectories[:, context_length:, -1].flatten())
        new_trajectories_confidence_score = torch.cat(new_trajectories_confidence_score, dim=0)
        if confidence:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories_confidence_score.sum(axis=-1))[:topk]]
        else:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories[..., context_length:, -1].sum(axis=-1))[-topk:]]
        print(new_trajectories.shape)

        optima = 1.0
        num_trajectories = self.dataset.num_trajectories + new_trajectories.shape[0]
        # print(self.dataset.points[0, :4, :10])
        # print(self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])[0, :4, :10])
        # print(self.dataset.values[0, -10:])
        # print(self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])[0, -10:])
        # print(kyle)
        points = torch.cat([self.dataset.points, self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])], dim=0)
        values = torch.cat([self.dataset.values, self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])], dim=0)
        
        # print(self.dataset.points[0, :10], new_trajectories[..., :-1][0, :10], self.dataset.normalizer.unnormalize(new_trajectories[..., :-1][0, :10]))
        # print(self.dataset.values[0, :10], new_trajectories[..., -1][0, :10], self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1][0, :10]))
        
        pointwise_regret = optima - values
        cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pointwise_regret, [1]), 1), [1])
        timesteps = torch.arange(horizon).repeat(num_trajectories, 1)
        
        self.dataset.num_trajectories = num_trajectories
        self.dataset.points = points
        self.dataset.values = values
        self.dataset.pointwise_regret = pointwise_regret
        self.dataset.cumulative_rtg = cumulative_regret_to_go
        self.dataset.timesteps = timesteps
        
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        ))
                
    def proxy_save(self):
        data = {
            'step': self.proxy_step,
            'model': self.proxy_model.state_dict(),
        }
        savepath = os.path.join(logger.prefix, 'proxy_checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.proxy_step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')
        
    def proxy_load(self, epoch=1000):
        loadpath = os.path.join(self.bucket, logger.prefix, f'proxy_checkpoint/state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)
        self.proxy_model.load_state_dict(data['model'])

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
