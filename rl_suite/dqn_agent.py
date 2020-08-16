import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .NN import ConvModel, MLP, IQN_MLP, IQNConvModel
from .agent import Agent

class DQN(Agent):
    def __init__(self, obs_shape, num_actions, modelpath, model_class=ConvModel, buffer_size=100000, discount = 0.99, lr=1e-4,
                 min_buffer_size=10000, total_episodes=0, training_steps=0, loss="huber", device=None, **kwargs):
        behaviour_model = model_class(obs_shape, num_actions)
        target_model = model_class(obs_shape, num_actions)
        target_model.eval()
        checkpoint = None
        if modelpath is not None:
            models = os.listdir(modelpath)
            if len(models):
                latest_model = max([int(name[:-4]) for name in models])
                checkpoint = modelpath/f'{latest_model}.pth'
        self.num_actions = num_actions
        if checkpoint is not None:
            saved = torch.load(checkpoint)
            print('loading checkpoint: ', checkpoint)
            behaviour_model.load_state_dict(saved['state_dict'])
            target_model.load_state_dict(saved['state_dict'])
            total_episodes = saved['total_episodes']
            training_steps = saved['training_steps']
        super().__init__(behaviour_model, target_model, modelpath=modelpath, buffer_size=buffer_size, discount=discount, lr=lr,
                         total_episodes=total_episodes, training_steps=training_steps, loss=loss, device=device, **kwargs)


    def train_step(self, data):
        states, actions, rewards, next_states, mask = self.make_data_tensors(data)
        self.opt.zero_grad()
        with torch.no_grad():
            if self.double_dqn:
                b_actions = self.model_b(next_states).max(-1)[1]
                # Getting contiguous array indexes
                idxs = b_actions + (self.num_actions * torch.arange(b_actions.shape[0]).to(self.device))
                qvals_next = self.model_t(next_states).take(idxs)
            else:
                qvals_next = self.model_t(next_states).max(-1)[0]
            G_t = rewards + (mask * self.discount * qvals_next)
            # G_t = rewards[:, 0] + (mask[:, 0] * self.discount * qvals_next)

        one_hot_actions = F.one_hot(actions, self.num_actions).to(self.device)
        qvals = self.model_b(states)
        if self.loss == "mse":
            Q_t = torch.sum(qvals * one_hot_actions, -1)
            td_error = (G_t - Q_t)
            loss = (td_error ** 2).mean()
        elif self.loss == "huber":
            Q_t = torch.sum(qvals * one_hot_actions, -1)
            td_error = (G_t - Q_t)
            loss = F.smooth_l1_loss(Q_t, G_t)

        loss.backward()
        self.opt.step()
        return loss.detach().item()

    def make_data_tensors(self, data):
        states = torch.tensor(data.state, device=self.device)
        # states = torch.tensor(data.state, dtype=torch.float, device=self.device) # LEAKS
        actions = torch.tensor(data.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(data.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(data.next_state, device=self.device)
        # next_states = torch.tensor(data.next_state, dtype=torch.float, device=self.device) # LEAKS
        mask = ~torch.tensor(data.done, dtype=torch.bool, device=self.device)
        return states, actions, rewards, next_states, mask

    def __call__(self, x, target=False):
        x = torch.tensor(x, device=self.device).unsqueeze(0)
        if target:
            qvals = self.model_t(x)
        else:
            qvals = self.model_b(x)
        return qvals






