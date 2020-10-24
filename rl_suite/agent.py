from collections import namedtuple
from dataclasses import dataclass
from typing import Any
import random
import numpy as np
import torch
import torch.optim as optim
import h5py

SARSD = namedtuple('SARSD', ('state', 'action', 'reward', 'next_state', 'done'))

@dataclass
class SARSD_:
    state: Any
    action: int
    reward: float
    next_state: Any
    done : bool


class ReplayBuffer:
    def __init__(self, buffer_size=100000, nstep_return=1, discount=None):
        self.buffer_size = buffer_size
        self.nstep_return = nstep_return
        if nstep_return > 1:
            assert discount is not None, "if nstep_return >1, discount must be specified"
        self.discount = discount
        # self.buffer = [None]*buffer_size
        self.buffer = []
        self.nstep_buffer = []
        self.idx=0

    def _add_to_replay(self, sarsd):
        if self.idx >= len(self.buffer):
            self.buffer.append(sarsd)
        else:
            self.buffer[self.idx] = sarsd
        self.idx = (self.idx + 1) % self.buffer_size

    def get_nstep_sarsd(self):
        _reward = sum([(self.discount**i)*dat.reward for i, dat
                        in enumerate(self.nstep_buffer)])
        nstep_sarsd = self.nstep_buffer.pop(0)
        nstep_sarsd.reward = _reward
        if len(self.nstep_buffer):
            nstep_sarsd.next_state = self.nstep_buffer[-1].next_state
        return nstep_sarsd

    def insert(self, sarsd):
        # Replay Buffer update generalized for multi-step setups
        if len(self.nstep_buffer) == self.nstep_return:
            self._add_to_replay(self.get_nstep_sarsd())
        self.nstep_buffer.append(sarsd)
        if sarsd.done:
            while len(self.nstep_buffer) > 0:
                self._add_to_replay(self.get_nstep_sarsd())

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

class ReplayBuffer_HDF5_v2(ReplayBuffer):
    """
    For use when not enough RAM is available to hold the whole buffer in memory
    I.e size 100,000 = 2 * (84 * 84 * 4) * 100,000/(1024**3) bytes = at least 5.3 GB of RAM
    To use compound dataset instead of separate datasets for each sarsd element
    """
    def __init__(self, example_item, buffer_size=100000, savepath=None, continue_exp=False, **params):
        self.buffer_size = buffer_size
        self.savepath = savepath
        # self.buffer = [None]*buffer_size
        self.buffer = []
        self._len = 0
        self.idx=0
        try:
            self.fields = tuple(example_item._fields)
        except:
            try:
                self.fields = tuple(example_item.__dataclass_fields.keys())
            except Exception as E:
                raise E
        if continue_exp:
            with h5py.File(savepath, 'r') as f:
                lens = set([len(f[item]) for item in self.fields])
                assert len(lens) == 1, "Lengths of each item in saved hdf5 buffer must be equal"
                self._len = f['current_idx'][0]
                self.idx = f['current_idx'][0]
        else:
            with h5py.File(savepath, 'w') as f:
                for field in self.fields:
                    ele = example_item.__getattribute__(field)
                    ele_shape = (buffer_size, ) + (ele.shape if isinstance(ele, np.ndarray) else (1, ))
                    dtype = ele.dtype if isinstance(ele, np.ndarray) else type(ele)
                    f.create_dataset(field, shape = ele_shape, dtype=dtype, chunks=True)
                f.create_dataset('current_idx', shape=(1,), dtype=int, data=[0])

    @property
    def current_idx(self):
        with h5py.File(self.savepath, 'r') as f:
            idx = f['current_idx'][0]
        return idx

    def insert(self, sars):
        # self.buffer.append(sars)
        with h5py.File(self.savepath, 'a') as f:
            for field in self.fields:
                f[field][self.idx, ...] = sars.__getattribute__(field)
        if self._len < self.buffer_size:
            self._len += 1
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, num_samples):
        assert num_samples < min(len(self), self.buffer_size), "Attempting to get sample of size larger than buffer"
        idxs = random.sample(range(len(self)), num_samples)
        idxs.sort() # sorted list required for indexing hdf5 datasets
        with h5py.File(self.savepath, 'r') as f:
            data = {}
            for field in self.fields:
                if field in ('reward', 'action', 'done'):
                    data[field] = f[field][idxs][:, 0]
                else:
                    data[field] = f[field][idxs]
        return SARSD(**data)

    def encode(self, sample):
        states = np.array([d.state for d in sample])
        actions = np.array([d.action for d in sample])
        rewards = np.array([d.reward for d in sample])
        next_states = np.array([d.next_state for d in sample])
        mask = np.array([[0] if d.done else 1 for d in sample])
        return SARSD(states, actions, rewards, next_states, done)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        return self.buffer[item]

    def __del__(self):
        with h5py.File(self.savepath, 'a') as f:
            f['current_idx'][0] = [self._len]

class Agent:
    def __init__(self, behaviour_model, target_model, modelpath, buffer_size=100000,
                 min_buffer_size=10000, example_obs=None, lr=1e-4, discount=0.99,
                 total_episodes=0, training_steps=0, device=None, loss='huber',
                 double_dqn=True, dueling=True, replay_period=4, batch_size=32,
                 nstep_return=None, continue_exp=False, buffer_savepath=None,
                 **params):
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
        self.replay_buffer = ReplayBuffer(buffer_size, nstep_return=nstep_return,
                                          discount=discount)
        self.lr = lr
        self.discount = discount
        self.opt = optim.Adam(self.model_b.parameters(), lr=self.lr)
        self.loss = loss
        self.total_episodes = total_episodes
        self.training_steps = training_steps
        self.double_dqn = double_dqn
        self.dueling = dueling
        assert nstep_return is not None, "Must explicitly specify nstep_return in agent constructor"
        self.nstep_return = nstep_return

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
