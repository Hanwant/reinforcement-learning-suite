import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

###########################################   Utility #####################################################
def conv_out_shape(in_shape, layers):
    """
    Calculates output shape of input_shape going through a list of pytorch convolutional layers
    in_shape: (H, W)
    layers: list of convolution layers
    """
    shape = in_shape
    for layer in layers:
        h_out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
        w_out = ((shape[1] + 2*layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1)-1) / layer.stride[1])+1
        shape = (int(h_out), int(w_out))
    return shape


####################################   DQN Base Architectures ############################################

class DuelingHead(nn.Module):
    """
    Replace normal output layer in DQN with this for Dueling Network Architectures
    See: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, d_model, num_actions, split=True):
        super().__init__()
        self.split=split
        if split:
            assert (d_model % 2 == 0)
            self.split_size = d_model // 2
            self.out_value = nn.Linear(self.split_size, 1)
            self.out_adv = nn.Linear(self.split_size, num_actions)
        else:
            self.out_value = nn.Linear(d_model, 1)
            self.out_adv = nn.Linear(d_model, num_actions)
    def forward(self, x):
        if self.split:
            x1, x2 = torch.split(x, self.split_size, -1)
        else:
            x1=x2=x
        vals = self.out_value(x1)
        adv = self.out_adv(x2)
        qvals = vals + adv - adv.mean(-1, keepdim=True) # broadcasts vals to add to adv for each sa pair
        return qvals


class MLP(nn.Module):
    def __init__(self, obs_shape, num_actions, d_model=256, dueling=False, nlayers=2, **params):
        super().__init__()
        self.in_shape = obs_shape[0]
        self.out_shape = num_actions
        self.d_model = d_model
        self.in_layer = nn.Sequential(nn.Linear(self.in_shape, d_model), nn.ReLU())
        layers = []
        for l in range(nlayers-1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.dueling = dueling
        if dueling:
            layers.append(DuelingHead(d_model, num_actions))
        else:
            layers.append(nn.Linear(d_model, num_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_layer(x.float())
        x = self.layers(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, dueling=True, discrete=False, reward_range=(-10, 10), natoms=51, d_model=512,
                 **params):
        super().__init__()
        self.in_shape = obs_shape
        self.out_shape = num_actions
        self.d_model = d_model
        self.c1 = nn.Conv2d(obs_shape[0], 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        out_shape = conv_out_shape((obs_shape[1], obs_shape[2]), [self.c1, self.c2, self.c3])
        self.fc = nn.Linear(out_shape[0]*out_shape[1]*64, d_model)
        self.dueling=dueling
        if dueling:
            self.out = DuelingHead(d_model, num_actions)
        else:
            self.out = nn.Linear(d_model, num_actions)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        """
        x: (B, C, H, W)
        out: (B, C, H, W)
        """
        x = self.act_fn(self.c1(x/255.0))
        x = self.act_fn(self.c2(x))
        x = self.act_fn(self.c3(x))
        x = self.act_fn(self.fc(x.view(x.shape[0], -1)))
        return self.out(x)

####################################   Architecures with Distributional Outputs  #####################################

class TauEmbedLayer(nn.Module):
    """
    For use in Implicit Quantile Networks (IQN) and Fully Parameterized Quantile Function (FQF)
    Takes quantile fractions as input and outputs an embedding
    to use for combining with state embedding and calculating
    quantile values

    see: https://arxiv.org//pdf/1806.06923.pdf and https://arxiv.org/pdf/1911.02140.pdf
    """
    def __init__(self, d_embed, d_model):
        super().__init__()
        self.d_embed = d_embed
        self.projection = nn.Linear(d_embed, d_model)
        self.act = nn.ReLU()

    def forward(self, tau):
        bs = tau.shape[0]
        N = tau.shape[1]
        # Embed using cosine function (I.e like in transformers)
        spectrum = math.pi * tau.view(bs, N, 1) * \
              torch.arange(1, self.d_embed+1, dtype=tau.dtype, device=tau.device).view(1, 1, self.d_embed)
        basis = torch.cos(spectrum)#.view(bs*N, self.d_embed) No Need to flatten as pytorch implicitly does it
        embedded = self.projection(basis)
        return self.act(embedded)

class IQN_MLP(nn.Module):
    def __init__(self, obs_shape, num_actions, d_model=256, d_embed=64, Ntau=32, dueling=False, nlayers=2, **params):
        super().__init__()
        self.in_shape = obs_shape[0]
        self.out_shape = num_actions
        self.d_model = d_model
        self.Ntau = Ntau
        self.in_layer = nn.Sequential(nn.Linear(self.in_shape, d_model), nn.ReLU())
        layers = []
        for l in range(nlayers-1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.tau_embed = TauEmbedLayer(d_embed, d_model)
        self.dueling = dueling
        if dueling:
            self.out_layer = DuelingHead(d_model, num_actions)
        else:
            self.out_layer = nn.Linear(d_model, num_actions)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, tau=None):
        if tau is None:
            tau = torch.rand(x.shape[0], self.Ntau, dtype=torch.float, device=x.device)
        assert tau.shape[0] == x.shape[0]
        bs = x.shape[0]
        state_embeddings = self.layers(self.in_layer(x.float()))
        tau_embeddings = self.tau_embed(tau)
        embeddings = (state_embeddings.unsqueeze(1) * tau_embeddings) #(bs, N, d_model)
        quantiles = self.out_layer(embeddings) # (bs, N, num_actions)
        return quantiles

class IQNConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, d_model=256, d_embed=64,
                 dueling=False, Ntau=32, **params):
        super().__init__()
        self.in_shape = obs_shape
        self.out_shape = num_actions
        self.d_model = d_model
        self.d_embed = d_embed
        self.dueling=dueling
        self.Ntau = Ntau
        self.c1 = nn.Conv2d(obs_shape[0], 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.act_fn = nn.ReLU()
        out_shape = conv_out_shape((obs_shape[1], obs_shape[2]), [self.c1, self.c2, self.c3])
        self.fc = nn.Linear(out_shape[0]*out_shape[1]*64, d_model)
        self.convlayers = nn.Sequential(self.c1, self.act_fn, self.c2, self.act_fn, self.c3, self.act_fn)
        self.tau_embed = TauEmbedLayer(d_embed, d_model)
        if dueling:
            self.out = DuelingHead(d_model, num_actions)
        else:
            self.out = nn.Linear(d_model, num_actions)

    def forward(self, x, tau=None):
        # assert tau is not None, "Must pass quantile fractions - tau"
        if tau is None:
            tau = torch.rand(x.shape[0], self.Ntau, dtype=torch.float, device=x.device)
        assert tau.shape[0] == x.shape[0]
        bs = x.shape[0]
        state_feats = self.convlayers(x/255.)
        state_embeddings = self.fc(state_feats.view(bs, -1))
        tau_embeddings = self.tau_embed(tau)
        embeddings = (state_embeddings.unsqueeze(1) * tau_embeddings) #(bs, N, d_model)
        quantiles = self.out(embeddings) # (bs, N, num_actions)
        return quantiles


class FractionProposalNetwork(nn.Module):
    def __init__(self, d_model, N):
        super().__init__()
        self.layer = nn.Linear(d_model, N)

    def forward(self, state_emb):
        q = F.softmax(self.layer(state_emb), -1)
        tau = torch.cumsum(q, -1)
        tau = torch.cat([torch.zeros((state_emb.shape[0], 1), dtype=state_emb.dtype, device=state_emb.device),
                        tau], dim=-1)
        tau_ = (tau[:, :-1] + tau[:, 1:]).detach() / 2.
        entropy = -torch.sum(q * torch.log(q), -1)
        return tau, tau_, entropy

class FQF_MLP(nn.Module):
    pass

class FQFConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, d_model=256, d_embed=64, dueling=False, Ntau=32, **params):
        super().__init__()
        self.in_shape = obs_shape
        self.out_shape = self.num_actions = num_actions
        self.d_model = d_model
        self.d_embed = d_embed
        self.dueling=dueling
        self.Ntau = Ntau
        self.c1 = nn.Conv2d(obs_shape[0], 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.act_fn = nn.ReLU()
        out_shape = conv_out_shape((obs_shape[1], obs_shape[2]), [self.c1, self.c2, self.c3])
        self.fc = nn.Linear(out_shape[0]*out_shape[1]*64, d_model)
        self.convlayers = nn.Sequential(self.c1, self.act_fn, self.c2, self.act_fn, self.c3, self.act_fn)
        self.tau_embed = TauEmbedLayer(d_embed, d_model)
        self.proposal_net = FractionProposalNetwork(d_model, Ntau)
        if dueling:
            self.out = DuelingHead(d_model, num_actions)
        else:
            self.out = nn.Linear(d_model, num_actions)

    def forward(self, x):
        # assert tau is not None, "Must pass quantile fractions - tau"
        state_embed = self.get_state_embeddings(x)
        tau, tau_, entropy = self.get_fractions(x=None, state_embed=state_embed)
        quantiles = self.get_quantiles(x=None, state_embed=state_embed, tau=tau)
        return quantiles, tau, tau_, entropy

    def get_fractions(self, x=None, state_embed=None):
        assert x is not None or state_embed is not None
        if state_embed is not None:
            bs = state_embed.shape[0]
            tau, tau_, entropy = self.proposal_net(state_embed.detach())
        else:
            state_embed = self.get_state_embeddings(x)
            tau, tau_, entropy = self.proposal_net(state_embed.detach())
        return tau, tau_, entropy


    def get_quantiles(self, x=None, state_embed=None, tau=None):
        assert x is not None or state_embed is not None
        if state_embed is None:
            state_embed = self.get_state_embeddings(x)
        if tau is None:
            tau, tau_, entropy = self.get_fractions(None, state_embed)

        tau_embed = self.tau_embed(tau)
        full_embed= state_embed.unsqueeze(1) * tau_embed
        quantiles = self.out(full_embed)
        return quantiles

    def get_state_embeddings(self, x):
        bs = x.shape[0]
        state_feats = self.convlayers(x/255.)
        return self.fc(state_feats.view(bs, -1))

    def get_q(self, x=None, state_embed=None, tau=None, tau_=None, quantiles_=None):
        assert x is not None or state_embed is not None or quantiles_ is not None

        if state_embed is None and quantiles_ is None:
            state_embed = self.get_state_embeddings(x)

        if tau is None or tau_ is None:
            tau, tau_, entropy = self.get_fractions(x=x, state_embed=state_embed)

        if quantiles_ is None:
            quantiles_ = self.get_quantiles(x=None, state_embed=state_embed, tau=tau_)
        assert quantiles_.shape[1:] == (self.Ntau, self.num_actions)

        qvals = ((tau[:, 1:, None] - tau[:, :-1, None])  * quantiles_).sum(dim=1)
        assert qvals.shape[1:] == (self.num_actions, )
        return qvals



if __name__ == "__main__":
    import gym
    from environments import AtariEnv
    print("testing on environment: Breakout")
    game = "Breakout-v0"
    env = AtariEnv(game, 84, 84)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    model = ConvModel(obs_shape, num_actions)
    im = env.reset()
    actions = model(torch.tensor(im, dtype=torch.float).unsqueeze(0))
    print("Q values: ", actions)
