from collections import namedtuple, defaultdict
import numpy as np
import random
import torch
import pdb
import pickle as pkl

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer, LimitsNormalizer, SafeLimitsNormalizer
from .buffer import ReplayBuffer

import design_bench
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    
TASKNAME2FULL = {
        'dkitty': DKittyMorphologyDataset,
        'ant': AntMorphologyDataset,
        'tfbind8': TFBind8Dataset,
        'tfbind10': TFBind10Dataset,
        'superconductor': SuperconductorDataset,
        }

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tfbind8': 'TFBind8-Exact-v0',
        'tfbind10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
}

TASKNAME2MAX_SAMPLES ={
    'dkitty': 10004,
    'ant': 10004,
    'tfbind8': 32898,
    'tfbind10': 50000,
    'superconductor': 17014,
}

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
    

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
    
    
class PointRegretDataset(torch.utils.data.Dataset):
    def __init__(self, horizon, data_path, context_length, regret=False, include_returns=False):
        with open(data_path, "rb") as f:
            points, values, pointwise_regret, cumulative_regret_to_go, timesteps = pkl.load(f)
    # def __init__(self, block_size, points, values, pointwise_regret, cumulative_rtg, timesteps, add_noise=False, variance=0.01):
        self.context_length = context_length
        self.block_size = horizon
        # self.vocab_size = args.vocab_size   # TODO
        self.num_trajectories = points.shape[0]
        self.size_of_trajectory = points.shape[1]
        self.points = points
        self.values = values
        self.pointwise_regret = pointwise_regret
        self.cumulative_rtg = cumulative_regret_to_go
        self.timesteps = timesteps
        self.regret = regret
        self.include_returns = include_returns

        # self.add_noise = add_noise
        # self.noise = torch.rand(self.num_trajectories, 1).repeat(1, self.size_of_trajectory) * variance

        # if self.add_noise:
        #     print("added noise: ", self.noise)
        self.observation_dim = self.points.shape[-1]
        self.action_dim = 1
        self.normalizer = SafeLimitsNormalizer(self.points)
        self.normalizer_values = SafeLimitsNormalizer(self.values.unsqueeze(-1))
        self.normalizer_values.mins = 0.0
        self.normalizer_values.maxs = 1.0
    
    def __len__(self):
        return self.num_trajectories * (self.size_of_trajectory - self.block_size + 1)

    def get_conditions(self, trajectories):
        conditions = {}
        conditions["ctx_len"] = np.array([0])
        return conditions

    def __getitem__(self, idx):
        traj_idx = idx // (self.size_of_trajectory - self.block_size + 1)
        ctx_idx = idx % (self.size_of_trajectory - self.block_size + 1)
        
        points = self.points[traj_idx, ctx_idx:ctx_idx+self.block_size] # (block_size, block_size, x_dim)
        if self.regret:
            values = self.pointwise_regret[traj_idx, ctx_idx:ctx_idx+self.block_size].unsqueeze(-1)
        else:
            values = self.values[traj_idx, ctx_idx:ctx_idx+self.block_size].unsqueeze(-1) # (block_size, block_size, 1

        points = self.normalizer.normalize(points)
        values_norm = self.normalizer_values.normalize(values)

        trajectories = torch.cat([points, values_norm], dim=-1)
        conditions = self.get_conditions(trajectories)
        
        if self.include_returns:
            returns = values.sum().unsqueeze(-1) / self.block_size
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)
        return batch
    

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, frac=1.0, sigma=0.0):
        task = design_bench.make(TASKNAME2TASK[dataset],
                                 dataset_kwargs=dict(
                                 max_samples=int(TASKNAME2MAX_SAMPLES[dataset] * frac),
                                 distribution=None,
                                 min_percentile=0)
                                )
        
        if dataset.startswith("tfbind"):
            task.map_to_logits()
        self.data_x = torch.from_numpy(task.x.reshape(task.x.shape[0], -1)).float()
        self.data_y = torch.from_numpy(task.y).float()
        
        fully_observed_task = TASKNAME2FULL[dataset]()
        self.data_y = (self.data_y - fully_observed_task.y.min()) / (fully_observed_task.y.max() - fully_observed_task.y.min())
        self.data_y = np.clip(self.data_y + np.random.randn(*self.data_y.shape) * sigma, 0.0, 1.0).float()
        
        self.normalizer = SafeLimitsNormalizer(self.data_x)
        
    def unnormalize_values(self, y):
        return y
        
    def __len__(self):
        return len(self.data_x)
        
    def __getitem__(self, idx):
        return self.normalizer.normalize(self.data_x[idx]), self.data_y[idx]
