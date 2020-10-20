# STL
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import random
from collections import deque

# Optional Tools for debugging.
# import psutil
# import gc
# import tracemalloc
# import memory_profiler
# from pympler import muppy, summary

# User Libraries
from tqdm import tqdm
import cv2
import click
import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt

# Current Module Imports
from rl_suite.environments import make_env
from rl_suite.agent import SARSD
from rl_suite.NN import ConvModel, MLP, IQN_MLP, IQNConvModel, FQF_MLP, FQFConvModel
from rl_suite.dqn_agent import DQN
from rl_suite.iqn_agent import IQN
# from rl_suite.fqf_agent import FQF
from rl_suite.utils import *

# will be imported inside function if send_to_wandb is set to True - so non-users don't have to install
# import wandb

logging.basicConfig(level=logging.INFO)

global device
global MLP_GAMES # Games for which a state vector is passed as input, instead of an image
MLP_GAMES = ('CartPole-v0', 'CartPole-v1')


def get_latest_id(basepath, game):
    try:
        exps = [int(exp) for exp in os.listdir(basepath/f'logs/{game}')]
        if len(exps):
            return str(max(exps))
        else:
            return None
    except FileNotFoundError:
        return "0"

def make_id(basepath, game, continue_exp=True):
    _id = ""
    recent = get_latest_id(basepath, game)
    if recent:
        if continue_exp:
            _id += recent
        else:
            _id += str(int(recent)+1)
    else:
        _id += "0"
    return _id

def setup_experiment(basepath, game, params, continue_exp=True, use_hdf5=False):

    exp_id = make_id(basepath, game, continue_exp=continue_exp)

    modelpath = basepath/'models'/game/exp_id
    imagepath = basepath/'images'/game/exp_id
    logpath = basepath/'logs'/game/exp_id
    parampath = basepath/'params'/game/exp_id
    bufferpath = basepath/'buffers'/game/exp_id

    for pth in (modelpath, imagepath, logpath, parampath, bufferpath):
        if not pth.is_dir():
            pth.mkdir(parents=True, exist_ok=True)
    logpath /= 'testlog.csv'
    parampath /= 'params.json'
    bufferpath /= 'replay_buffer.hdf5'

    if not parampath.exists():
        with open(parampath, 'w') as f:
            json.dump(params, f)
    else:
        with open(parampath, 'r') as f:
            params = json.load(f)

    return exp_id, logpath, parampath, modelpath, imagepath, bufferpath, params

def init_wandb(game, exp_id, params, continue_exp=False):
    import wandb
    wandb.init(project=f"{game}_exp_{exp_id}", name=game, resume=continue_exp)
    wandb.config.buffer_size = params['buffer_size']
    wandb.config.min_buffer_size = params['min_buffer_size']
    wandb.config.batch_size = params['batch_size' ]
    wandb.config.replay_period = params['replay_period']
    wandb.config.save_period = params['save_period']
    wandb.config.lr = params['lr']
    wandb.config.expl = params['expl']

def init_logs(agent, env, logpath, send_to_wandb=False):
    if not logpath.exists():
        episode_length, rewards, action_values, frames = run_test_episode(agent, env, device)
        logs = pd.DataFrame({'updated': [datetime.now()], 'training_steps': [0], 'episode_length': [episode_length],
                             'reward': [rewards], 'loss': [0.]})
        logs.to_csv(logpath, sep=',')
    if send_to_wandb:
        init_wandb(agent.game, exp_id, params)

def update_logs(logpath, loss, rewards, total_steps, episode_length, training_steps, send_to_wandb = False, videopath=None):
    log = pd.read_csv(logpath, sep=',', header=[0], index_col=0)
    log = log.append(pd.DataFrame({'updated': [datetime.now()], 'total_steps': [total_steps],
                                   'training_steps': [training_steps], 'episode_length': [episode_length],
                                   'loss': [loss], 'reward': [rewards]}))
    log.to_csv(logpath, sep=',', header=True)
    if send_to_wandb:
        if videopath is not None:
            wandb.log({'loss': loss, 'reward': reward, 'video': wandb.Video(videopath)}, step = total_steps)
        else:
            wandb.log({'loss': loss, 'reward': reward}, step = total_steps)

def run_test_episode(agent, env, device, max_steps=1000, eps=0.1, init_steps=4, boltzmann=False, boltzmann_temp = 0.8): # -> reward, log/movie
    frames = []
    obs = env.reset()
    frame = env.frame
    idx = 0
    done = False
    reward = 0
    action_values = []
    # print(init_steps)
    while not done and idx < max_steps:
        if init_steps > 0:
            action = env.action_space.sample()
            init_steps -= 1
        else:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action_vals = agent(obs, target=True)
                    action_values.append((obs, action_vals.cpu().numpy()))
                    if boltzmann:
                        action = torch.distributions.Categorical(logits = action_vals/boltzmann_temp).sample()
                    else:
                        action = action_vals.max(-1)[1].item()
        # print(action)
        obs, r, done, _ = env.step(action)
        reward += r
        frames.append(env.render(mode="rgb_array"))
        idx += 1
    if idx == max_steps:
        print(f'max testing steps reached ({idx})')
    env.close()
    return idx, reward, action_values, np.stack(frames, 0)


def show_action_values(action_values, idxs):
    fig, axs = plt.subplots(1, len(idxs))
    for i in range(len(idxs)):
        axs[i].imshow(action_values[idxs[i]][0][0], cmap="Greys")
        txt = '\n '.join([f'action {a}: {action_values[idxs[i]][1][0][a]: .3f}' for a in range(len(action_values[0][1][0]))])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[i].text(0.05, 0.05, txt, transform=axs[i].transAxes, fontsize=7,
                verticalalignment='bottom', bbox=props)
    plt.show()



def train_loop(agent, env, basepath, exp_id, send_to_wandb=False, replay_period=4, save_period=32000, max_steps=100000, expl="eps", nstep=1, eps=1., eps_decay=1e-5, eps_min=0.1, frame_skip=4, boltzmann_temp = 0.8, max_test_steps=10000, **kwargs):
    # tracemalloc.start()
    # snap1 = tracemalloc.take_snapshot( )
    assert expl in ('eps', 'boltzmann')

    logpath = basepath/f'logs/{env.game}/{exp_id}/testlog.csv'
    imagepath = basepath/f'images/{env.game}/{exp_id}/'
    init_logs(agent, env, logpath, send_to_wandb=send_to_wandb)

    i = agent.total_episodes
    max_steps += i
    buffer_size = agent.buffer_size
    replay_period = agent.replay_period
    rb = agent.replay_buffer
    nstep_buffer = []
    steps_since_buffer_update = 0
    save_period = save_period
    steps_since_t_update = 0
    training_steps = agent.training_steps
    rolling_reward = 0.
    last_episode_reward = 0.
    test_reward = 0.
    loss = 0.

    eps = eps * (1-eps_decay) ** i
    state = env.reset()
    # episode_rewards = []
    try:
        tq=tqdm(total=max_steps)
        tq.update(i)
        while True:
            tq.update(1)
            ##################################### HOUSE KEEPING ########################################
            # Initial Logging of reward, loss
            if i == 0:
                episode_length, reward, action_values, frames = run_test_episode(agent, env, agent.device)
                update_logs(logpath, 0., reward, i, episode_length, training_steps, send_to_wandb=False)

            # Max steps
            if i > max_steps:
                print(f'Max Steps reached ({max_steps}) \nExiting...')
                break

            ##################################### AGENT DECIDES ON ACTION ########################################
            # Atari Games already implement frame skip of 4. So here, frame_skip only needs to be 1
            if i % frame_skip == 0:
                # Get action with respect to given scheme
                if expl == "eps":
                    if random.random() < eps:
                        action = env.action_space.sample()
                    else:
                        action = agent(state, target=False).max(-1)[-1].item()
                elif expl == 'boltzmann':
                    qvals = agent(state, target=False) / boltzmann_temp
                    action = torch.distributions.Categorical(logits = qvals).sample().item()

            eps = max(eps*(1-eps_decay), eps_min)

            ##################################### ENVIRONMENT RESPONDS ########################################
            # Take action, get SARSD
            new_state, reward, done, info = env.step(action)
            rolling_reward += reward

            # Replay Buffer update generalized for multi-step setups
            nstep_buffer.append(SARSD(state, action, reward, new_state, done))
            if len(nstep_buffer) == nstep:
                _reward = sum([dat.reward for dat in nstep_buffer])
                sarsd = nstep_buffer.pop(0)
                nstep_sarsd = SARSD(sarsd.state, sarsd.action, _reward, sarsd.next_state, sarsd.done)
                agent.replay_buffer.insert(nstep_sarsd)
                # if nstep == 1:
                #     assert all(np.allclose(orig, _nstep) for orig, _nstep in zip(sarsd, nstep_sarsd))


            steps_since_buffer_update += 1
            state = new_state

            # If episode is done, reset environment and rolling reward
            if done:
                last_episode_reward = rolling_reward
                rolling_reward = 0.
                state = env.reset()

            ##################################### AGENT LEARNS   ###############################################
            # Criteria for training
            if len(rb) >= agent.min_buffer_size and i % agent.replay_period == 0:
                data = agent.replay_buffer.sample(agent.batch_size)
                # tot1=mem()
                loss = agent.train_step(data)
                # tot2=mem()
                # print('tot1: ', tot1)
                # print('tot2: ', tot2)
                # print('diff: ', tot2-tot1)
                if send_to_wandb:
                    wandb.log({'loss': loss, 'last_reward': last_episode_reward}, step=i)
                # Criteria for updating target model and saving model+logs+video
                if steps_since_t_update > save_period:
                    agent.model_t.load_state_dict(agent.model_b.state_dict())
                    agent.training_steps += 1
                    agent.save_model(total_episodes=i, training_steps=agent.training_steps)
                    episode_length, test_reward, action_values, frames = run_test_episode(agent, env, agent.device, max_steps=max_test_steps)
                    make_video(frames, imagepath/f'step_{i}_reward_{test_reward}.mp4')
                    update_logs(logpath, loss, test_reward, i, episode_length, agent.training_steps,
                              send_to_wandb=send_to_wandb, videopath=imagepath/f'step_{i}_reward_{test_reward}.mp4')
                    steps_since_t_update = 0

            if i % 20000 == 0:
                print(f'training buffer_size: {len(rb)} buffer_idx: {rb.idx} last_training_reward: {last_episode_reward} last_test_reward: {test_reward} loss: {loss}',
                    f'eps: {eps}')
            i += 1
            steps_since_t_update += 1

    except KeyboardInterrupt:
        # current, peak = tracemalloc.get_traced_memory()
        # snap2 = tracemalloc.take_snapshot()
        # stats = snap2.compare_to(snap1, 'filename')
        import ipdb; ipdb.set_trace()
        # all_obj = muppy.get_objects()
        # sum1 = summary.summarize(all_obj)
        # summary.print_(sum1)
        # for i, stat in enumerate(stats[:5], 1):
        #     print(f"{i}. since last snap stat: ", str(stat))
        # pass
    except Exception as E:
        raise E

@click.command()
@click.argument('game', default = 'CartPole-v0')
@click.option('--train/--test', default = True, help='Choose from: --test, --train')
@click.option('--continue_exp/--new_exp', default = True, help='Continue from last saved experiment or start new')
@click.option('--max_steps', default = 1000000, help='Max number of steps before terminating training loop')
@click.option('--max_test_steps', default = 10000, help='Max number of test steps before terminating episode')
@click.option('--send_to_wandb', default = False, help='Init and log to wandb')
@click.option('--use_cuda', default = True  if torch.cuda.is_available() else False, help='Set to True to use cuda (if enabled)')
def main(game, train, continue_exp, max_steps, send_to_wandb, max_test_steps, use_cuda):
    basepath = Path(os.getcwd())
    if not train: # if just testing then the parameters and weights will be loaded from the previous experiment
        continue_exp = True
    global device
    if use_cuda:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    # DEFAULT PARAMS IF param.json file is not found in parampath for experiment
    params = dict(double_dqn=True,
                  dueling=True,
                  prioritized_replay=True,
                  multi_step=True,
                  nstep=3,
                  noisy_nets=True,

                  IQN=True,
                  FQF=False,
                  d_embed=64,
                  Ntau=32,
                  Ntau1=32,
                  Ntau2=8,
                  lr_fraction_net=2.5e-9,
                  entropy_coeff=0.,

                  buffer_size=100000,
                  min_buffer_size=50000,
                  use_hdf5=False,
                  buffer_savepath=None,
                  batch_size=32,
                  replay_period=4,
                  save_period=10000,

                  d_model=512,
                  lr=1e-4,
                  discount=0.99,
                  loss='huber',
                  k_huber=1.,

                  expl='eps',
                  eps=1.,
                  eps_decay=1e-6,
                  eps_min=0.1,

                  frame_skip=1) # Should only be needed for custom environments


    exp_id, logpath, parampath, modelpath, imagepath, bufferpath, params = setup_experiment(basepath, game, params, continue_exp=continue_exp)
    if params['buffer_savepath'] is None:
        params['buffer_savepath'] = bufferpath

    logging.info(f"""game: {game} exp_id: {exp_id} \nlogpath: {logpath} \nparampath: {parampath}
                   \nmodelpath: {modelpath} \nimagepath: {imagepath} \nparams: {params} \n
    {'buffer savepath: ' + str(params['buffer_savepath']) if params['use_hdf5'] else None}""")

    assert not (params['IQN'] and params['FQF']), "Can only choose 1 Distributional Approach out of (IQN, FQF)"
    if game in MLP_GAMES:
        if params['IQN']:
            model_class = IQN_MLP
        elif params['FQF']:
            # model_class = FQF_MLP
            raise NotImplementedError('FQF not implemented yet')
        else:
            model_class = MLP
        params['frame_skip'] = 1
    else:
        if params['IQN']:
            model_class = IQNConvModel
        elif params['FQF']:
            # model_class = FQFConvModel
            raise NotImplementedError('FQF not implemented yet')
        else:
            model_class = ConvModel


    env = make_env(game, 84, 84)
    # example obs -> sarsd used when initializing hdf5 dataset
    obs = env.reset()
    sarsd = SARSD(state=obs, action=0, reward=0., next_state=obs, done=True)
    env = make_env(game, 84, 84)

    if params['IQN']:
        agent = IQN(env.observation_space.shape, env.action_space.n, modelpath, continue_exp=continue_exp,
                    model_class=model_class, example_obs=sarsd, device=device, **params)
    elif params['FQF']:
        agent = FQF(env.observation_space.shape, env.action_space.n, modelpath, continue_exp=continue_exp,
                    model_class=model_class, example_obs=sarsd, device=device, **params)
    else:
        agent = DQN(env.observation_space.shape, env.action_space.n, modelpath, continue_exp=continue_exp,
                    model_class=model_class, example_obs=sarsd, device=device, **params)

    if train:
        train_loop(agent, env, basepath, exp_id, send_to_wandb=send_to_wandb, max_steps=max_steps, max_test_steps=max_test_steps, **params)

    else:
        episode_length, reward, action_values, frames = run_test_episode(agent, env, device, max_steps=max_test_steps)
        print("reward: ", reward)
        print("Saving test video to ", imagepath/"test_colour.mp4")
        make_video(frames, imagepath/"test_colour.mp4")
        model_input = np.vstack([frame[0][::-1, :, :] for frame in action_values])[:, :, :, None]
        print(model_input.shape)
        print("Saving test video to ", imagepath/"test_prepro.mp4")
        make_video(model_input, imagepath/"test_prepro.mp4")


if __name__ == "__main__":
    main()


