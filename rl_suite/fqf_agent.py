import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .NN import ConvModel, MLP, FQF_MLP, FQFConvModel
from .agent import Agent
from .dqn_agent import DQN
from .utils import huber_quantile_loss



class FQF(Agent):
    def __init__(self, obs_shape, num_actions, modelpath, d_embed, model_class=FQFConvModel, Ntau=32,
                 k_huber=1., lr_fraction_net = 2.5e-9, entropy_coeff=0., risk_distortion = lambda x: x, **kwargs):
        assert model_class in (FQF_MLP, FQFConvModel)
        behaviour_model = model_class(obs_shape, num_actions, d_embed=d_embed, Ntau=Ntau, **kwargs)
        target_model = model_class(obs_shape, num_actions, d_embed=d_embed, Ntau=Ntau, **kwargs)
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
        super().__init__(behaviour_model, target_model, modelpath=modelpath, **kwargs)
        self.Ntau = Ntau
        self.k_huber = k_huber
        self.entropy_coeff = entropy_coeff
        self.risk_distortion = risk_distortion
        self.optim = None
        self.optim_quantile = optim.Adam(list(self.model_b.convlayers.parameters()) + \
                                         list(self.model_b.tau_embed.parameters()) + \
                                         list(self.model_b.out.parameters()), lr=self.lr)
        self.optim_fraction = optim.RMSprop(list(self.model_b.proposal_net.parameters()),
                                            lr = lr_fraction_net, alpha=0.95, eps=1e-5)


    def train_step(self, data):
        states, actions, rewards, next_states, mask = self.make_data_tensors(data)

        one_hot_actions = F.one_hot(actions[:, None], self.num_actions).to(self.device)
        state_embed = self.model_b.get_state_embeddings(states)
        tau, tau_, entropy = self.model_b.get_fractions(x=None, state_embed=state_embed)
        quantiles_current_ = self.model_b.get_quantiles(x=None, state_embed=state_embed, tau=tau_)
        assert quantiles_current_.shape[1:] == (self.Ntau, self.num_actions)
        Qt_ = (quantiles_current_ * one_hot_actions).sum(-1)
        # qvals_current = ((tau[:, 1:, None] - tau[:, :-1, None]) * quantiles_current_).sum(1)

        with torch.no_grad():
            if self.double_dqn:
                b_actions = self.model_b(next_states).max(-1)[1]
                idxs = b_actions + (self.num_actions * torch.arange(b_actions.shape[0]).to(self.device))
                quantiles_next = self.model_t(next_states, tau=tau2).take(idxs)
            else:
                next_state_embed = self.model_t.get_state_embeddings(next_states)
                # next_q calclated by target model uses its own tau and tau_ when not passed any as args
                next_q = self.model_t.get_q(x=None, state_embed=next_state_embed)
                # greedy_quantiles = self.model_t.get_quantiles(x=None, state_embed = next_state_embed,
                #                                                tau=tau_) #(bs, quantiles, num_actions)
                # next_qvals = self.model_t.get_q(x=None, state_embed=None, tau=tau, tau_=tau_,
                #                                 quantiles_=greedy_quantiles)
                next_actions = torch.argmax(next_q, dim=-1, keepdim=True)
                next_greedy_actions = F.one_hot(next_actions, self.num_actions).to(self.device)
                quantiles_next_ = self.model_t.get_quantiles(x=None, state_embed=next_state_embed, tau=tau_)
                quantiles_next_ = (quantiles_next_ * next_greedy_actions).sum(-1)
                Gt_ = rewards[:, None] + self.discount * mask[:, None] * quantiles_next_

        td_errors = Gt_.unsqueeze(1) - Qt_.unsqueeze(-1) # (bs, Ntau, Ntau)
        quantile_loss = self.quantile_huber_loss(td_errors, tau_)
        fraction_loss = self.fraction_loss(state_embed.detach(), Qt_.detach(), tau, actions)
        entropy_loss = self.entropy_coeff * entropy.mean()
        fraction_loss += entropy_loss

        self.optim_fraction.zero_grad()
        self.optim_quantile.zero_grad()
        fraction_loss.backward(retain_graph=True)
        quantile_loss.backward()
        self.optim_fraction.step()
        self.optim_quantile.step()

        return (fraction_loss + quantile_loss).detach().item()

    def quantile_huber_loss(self, tderr, tau):
        """
        tderr: td errors (bs, Ntau, Ntau)
        """
        assert tderr.shape[1:] == (self.Ntau, self.Ntau), "td errors must be of shape (bs, Ntau, Ntau)"
        huber_loss = torch.where(tderr.abs() <= self.k_huber,
                                 0.5*tderr.pow(2),
                                 self.k_huber*(tderr.abs() - self.k_huber/2))
        assert huber_loss.shape == tderr.shape
        quantile_loss = torch.abs(tau[..., None] - (tderr.detach() < 0.).float()) \
                        * huber_loss / self.k_huber
        assert quantile_loss.shape == huber_loss.shape
        return quantile_loss.mean(-1).sum(-1).mean()

    def fraction_loss(self, state_embed, Qt_, tau, actions):
        """
        Fraction Loss
        """
        with torch.no_grad():
            Qt = self.model_b.get_quantiles(x=None, state_embed=state_embed, tau=tau[:, 1:-1])
            one_hot_actions = F.one_hot(actions[:, None], self.num_actions).to(self.device)
            Qt = (Qt * one_hot_actions).sum(-1)
            # Qt = Qt_[:, 1:-1]
        vals1 = Qt - Qt_[:, :-1]
        signs1 = Qt > torch.cat([Qt_[:, :1], Qt[:, :-1]], dim=1)
        vals2 = Qt - Qt_[:, 1:]
        signs2 = Qt < torch.cat([Qt[:, 1:], Qt_[:, -1:]], dim=1)
        fraction_gradient = (torch.where(signs1, vals1, -vals1) +\
                         torch.where(signs2, vals2, -vals2)
                         ).view(state_embed.shape[0], self.Ntau-1)
        assert not fraction_gradient.requires_grad
        fraction_loss = (fraction_gradient * tau[:, 1:-1]).sum(dim=1).mean()
        return fraction_loss


    def huber_loss(self, tderr):
        """
        Calculates Huber Loss, no reduction
        """
        return torch.where(tderr.abs() <= self.k_huber, 0.5*tderr.pow(2),
                    self.k_huber*(tderr.abs() - self.k_huber/2))

    def quantile_loss(self, tderr, huber_loss, tau):
        """
        tderr: shape (bs, Ntau-1, Ntau-1)
        huber_loss: shape (bs, Ntau-1, Ntau-1)
        tau: shape (bs, Ntau)
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
            qvals = self.model_t.get_q(x) #(bs, Ntau, num_actions)
        else:
            qvals = self.model_b.get_q(x) #(bs, Ntau, num_actions)
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
    agent = FQF(obs_shape, 6, None, d_embed=64, d_model=256, double_dqn=False, dueling=False,
                Ntau=28)
    rb = agent.replay_buffer

    for i in range(100):
        rb.insert(make_data())

    data = rb.sample(32)
    print("================  TESTING ================\n\n")
    print("Starting learning loop")
    print("Training repeatedly on single batch to confirm learning capacity \n")
    for i in range(2000):
        loss = agent.train_step(data)
        if i % 100 == 0:
            print(f"step: {i} loss:", loss)















