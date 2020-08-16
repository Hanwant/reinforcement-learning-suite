import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .NN import ConvModel, MLP, IQN_MLP, IQNConvModel
from .agent import Agent
from .dqn_agent import DQN
from .utils import huber_quantile_loss



class IQN(Agent):
    def __init__(self, obs_shape, num_actions, modelpath, d_embed, model_class=IQNConvModel, Ntau1=32, Ntau2=32,
                 k_huber=1., risk_distortion = lambda x: x, **kwargs):
        assert model_class in (IQN_MLP, IQNConvModel)
        behaviour_model = model_class(obs_shape, num_actions, d_embed=d_embed, **kwargs)
        target_model = model_class(obs_shape, num_actions, d_embed=d_embed, **kwargs)
        target_model.eval()
        self.Ntau1 = Ntau1
        self.Ntau2 = Ntau2
        self.k_huber = k_huber
        self.risk_distortion = risk_distortion
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
        super().__init__(behaviour_model, target_model, modelpath=modelpath, **kwargs)

    def train_step(self, data):
        states, actions, rewards, next_states, mask = self.make_data_tensors(data)
        # quantiles fractions for getting the next greedy action
        tau_b = torch.rand(states.shape[0], self.Ntau1, dtype=torch.float, device=self.device)
        tau_b = self.risk_distortion(tau_b)
        # quantiles for current qvals
        tau1 = torch.rand(states.shape[0], self.Ntau1, dtype=torch.float, device=self.device)
        #  quantiles for next qvals (given greedy action)
        tau2 = torch.rand(states.shape[0], self.Ntau2, dtype=torch.float, device=self.device)

        self.opt.zero_grad()
        with torch.no_grad():
            if self.double_dqn:
                greedy_quantiles = self.model_b(next_states, tau=tau_b) #(bs, quantiles, num_actions)
                next_actions = torch.argmax(greedy_quantiles.mean(dim=1), dim=-1, keepdim=True)
                assert next_actions.shape == (self.batch_size, 1)
                one_hot_actions = F.one_hot(next_actions, self.num_actions).to(self.device)
                quantiles_next = self.model_t(next_states, tau=tau2)
                qvals_next = (quantiles_next * one_hot_actions).sum(-1)
            else:
                greedy_quantiles = self.model_t(next_states, tau=tau_b) #(bs, quantiles, num_actions)
                next_actions = torch.argmax(greedy_quantiles.mean(dim=1), dim=-1, keepdim=True)
                assert next_actions.shape == (self.batch_size, 1)
                one_hot_actions = F.one_hot(next_actions, self.num_actions).to(self.device)
                quantiles_next = self.model_t(next_states, tau=tau2)
                qvals_next = (quantiles_next * one_hot_actions).sum(-1)
            G_t = rewards[:, None] + self.discount * mask[:, None] * qvals_next

        one_hot_actions = F.one_hot(actions[:, None], self.num_actions).to(self.device)
        quantiles_current = self.model_b(states, tau=tau1)
        Q_t = (quantiles_current* one_hot_actions).sum(-1)
        td_errors = G_t.unsqueeze(1) - Q_t.unsqueeze(-1) # (bs, Ntau1, Ntau2)
        loss = self.quantile_huber_loss(td_errors, tau1)
        loss.backward()
        self.opt.step()
        return loss.detach().item()

    def quantile_huber_loss(self, tderr, tau):
        """
        tderr: td errors (bs, Ntau1, Ntau2)
        """
        assert tderr.shape[1:] == (self.Ntau1, self.Ntau2), "td errors must be of shape (bs, Ntau1, Ntau2)"
        huber_loss = torch.where(tderr.abs() <= self.k_huber,
                                 0.5*tderr.pow(2),
                                 self.k_huber*(tderr.abs() - self.k_huber/2))
        assert huber_loss.shape == tderr.shape
        quantile_loss = torch.abs(tau[..., None] - (tderr.detach() < 0.).float()) \
                        * huber_loss / self.k_huber
        assert quantile_loss.shape == huber_loss.shape
        return quantile_loss.mean(-1).sum(-1).mean()

    def huber_loss(self, tderr):
        """
        Calculates Huber Loss, no reduction
        """
        return torch.where(tderr.abs() <= self.k_huber, 0.5*tderr.pow(2),
                    self.k_huber*(tderr.abs() - self.k_huber/2))

    def quantile_loss(self, tderr, huber_loss, tau):
        """
        tderr: shape (bs, Ntau1, Ntau2)
        huber_loss: shape (bs, Ntau1, Ntau2)
        tau: shape (bs, Ntau1)
        """
        return torch.abs(tau[..., None] - (tderr.detach() < 0.).float()) * huber_loss / self.k_huber

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
        x = torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(0)
        if target:
            quantiles = self.model_t(x) #(bs, Ntau, num_actions)
            qvals = quantiles.mean(1)
        else:
            quantiles = self.model_b(x) #(bs, Ntau, num_actions)
            qvals = quantiles.mean(1)
        return qvals

if __name__ == "__main__":
    import numpy as np
    from Agent import SARSD, ReplayBuffer

    def make_data():
        state = np.random.normal(0, 1, (4, 84, 84)).astype(np.float32)
        action = np.random.randint(0, 6, 1).item()
        reward = np.random.normal(0, 1, 1).astype(np.float32).item()
        next_state = np.random.normal(0, 1, (4, 84, 84)).astype(np.float32)
        done = np.random.binomial(1, 0.5, 1).astype(np.bool).item()
        return SARSD(state, action, reward, next_state, done)

    obs_shape = make_data().state.shape
    agent = IQN(obs_shape, 6, None, d_embed=64, d_model=256, double_dqn=False, dueling=False,
                Ntau1=28, Ntau2=36)
    rb = agent.replay_buffer

    for i in range(100):
        rb.insert(make_data())

    data = rb.sample(32)
    print("================  TESTING ================\n\n")
    print("Starting learning loop")
    print("Training repeatedly on single batch to confirm learning capacity \n")
    for i in range(1000):
        loss = agent.train_step(data)
        if i % 100 == 0:
            print(f"step: {i} loss:", loss)















