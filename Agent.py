from dataclasses import dataclass
from typing import Any
import random
import numpy as np
import torch
import torch.optim as optim
from collections import namedtuple

SARSD = namedtuple('SARSD', ('state', 'action', 'reward', 'next_state', 'done'))

@dataclass
class SARSD_:
    state: Any
    action: int
    reward: float
    next_state: Any
    done : bool


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        # self.buffer = [None]*buffer_size
        self.buffer = []
        self.idx=0

    def insert(self, sars):
        # self.buffer.append(sars)
        if self.idx >= len(self.buffer):
            self.buffer.append(sars)
        else:
            self.buffer[self.idx] = sars
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, num_samples):
        assert num_samples < min(len(self), self.buffer_size)
        if len(self)< self.buffer_size:
            return self.encode(random.sample(self.buffer[:self.idx], num_samples))
        return self.encode(random.sample(self.buffer, num_samples))

    def encode(self, sample):
        states = np.stack([d.state for d in sample], axis=0)
        actions = np.array([d.action for d in sample], dtype=np.int)
        rewards = np.array([d.reward for d in sample], dtype=np.float)
        next_states = np.stack([d.next_state for d in sample], axis=0)
        done = np.array([d.done for d in sample], dtype=np.bool)
        return SARSD(states, actions, rewards, next_states, done)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

class ReplayBuffer_HDF5(ReplayBuffer):
    """
    For use when not enough RAM is available to hold the whole buffer in memory
    I.e 100,000 = 2 * (84 * 84 * 4 * 8) * 10000/1000000000 bytes = 11.3 GB
    """
    def __init__(self, example_item, buffer_size=100000, savepath=None, resume=False, **params):
        import h5py
        self.buffer_size = buffer_size
        # self.buffer = [None]*buffer_size
        self.buffer = []
        self.idx=0
        self.fields = tuple(example_item.__dataclass_fields__.keys())
        if resume:
            with h5py.File(savepath, 'r') as f:
                lens = set([len(f[item]) for item in self.fields])
                assert len(lens) == 1, "Lengths of each item in saved hdf5 buffer must be equal"
        else:
            with h5py.File(savepath, 'w') as f:
                for field in self.fields:
                    ele = example_item.__getattribute(field)
                    ele_shape = (buffer_size, ) + ele.shape if isinstance(ele, np.ndarray) else (len(ele), )
                    f.create_dataset(field, shape = ele_shape, dtype=ele.dtype, chunks=True)

    def insert(self, sars):
        # self.buffer.append(sars)
        if self.idx >= len(self.buffer):
            self.buffer.append(sars)
        else:
            self.buffer[self.idx] = sars
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, num_samples):
        assert num_samples < min(len(self), self.buffer_size)
        if len(self)< self.buffer_size:
            return random.sample(self.buffer[:self.idx], num_samples)
        return random.sample(self.buffer, num_samples)

    def encode(self, sample):
        states = np.array([d.state for d in sample])
        actions = np.array([d.action for d in sample])
        rewards = np.array([d.reward for d in sample])
        next_states = np.array([d.next_state for d in sample])
        mask = np.array([[0] if d.done else 1 for d in sample])

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

class Agent:
    def __init__(self, behaviour_model, target_model, modelpath, buffer_size=100000, min_buffer_size=10000, lr=1e-4, discount=0.99,
                 total_episodes=0, training_steps=0, device=None, loss='huber', double_dqn=True, dueling=True, replay_period=4,
                 batch_size=32, **params):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_b = behaviour_model.to(self.device)
        self.model_t = target_model.to(self.device)
        self.model_t.eval()
        for param in self.model_t.parameters():
            param.reqiures_grad=False
        self.modelpath = modelpath
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.replay_period = replay_period
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.lr = lr
        self.discount = discount
        self.opt = optim.Adam(self.model_b.parameters(), lr=self.lr)
        self.loss = loss
        self.total_episodes = total_episodes
        self.training_steps = training_steps
        self.double_dqn = double_dqn
        self.dueling = dueling

    def train_step(self, data):
        raise NotImplementedError

    def update_buffer(self, sars):
        self.replay_buffer.insert(sars)

    def save_model(self, total_episodes, training_steps):
        to_save = {'state_dict': self.model_t.state_dict(), 'total_episodes': total_episodes,
                   'training_steps': training_steps}
        torch.save(to_save, self.modelpath/f'{total_episodes}.pth')

    def __call__(self, x, target=False):
        x = torch.tensor(obs, device=self.device).unsqueeze(0)
        if target:
            qvals = self.model_t(x)
        else:
            qvals = self.model_b(x)
        return qvals
