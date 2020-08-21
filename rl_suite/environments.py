import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gym
from dataclasses import dataclass
"""

env.reset() -> obs
env.step(action: int) -> obs: ndarray, reward: float, done: bool, info: Dict
env.render

"""

games = ()


def make_env(game, width=80, height=110, num_stack=4):
    if game in ('CartPole-v0', 'CartPole-v1'):
        return CartPoleEnv(game)
    else:
        return AtariEnv(game, width=width, height=height, num_stack=num_stack)

class CartPoleEnv:
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        self.frame = None
        self.dtype = np.float32

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode=None):
        # if mode == "rgb_array":
        #     return self.frame
        # else:
        self.frame = self.env.render(mode="rgb_array")
        return self.frame

    def reset(self):
        obs = self.env.reset()
        # self.frame = self.env.render(mode="rgb_array")
        return obs

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        # self.frame = self.env.render(mode="rgb_array")
        return np.array(obs, dtype=self.dtype), reward, done, info

    def close(self):
        self.env.close()


class AtariEnv:
    def __init__(self, game, width, height, num_stack=4):
        self.game = game
        self.env = gym.make(game)
        self.n = num_stack
        self.w = width
        self.h = height
        self.buffer = np.zeros((num_stack, self.h, self.w), 'uint8')
        self.frame=None


    @property
    def observation_space(self):
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space

    def _preprocess_frame(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def render(self, mode=None):
        if mode == "rgb_array":
            return self.frame
        else:
            self.env.render()

    def reset(self):
        im = self.env.reset()
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im]*self.n, 0)
        return self.buffer.copy()
        # return self.buffer

    def step(self, action: int):
        im, reward, done, info = self.env.step(action)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer[1: self.n, :, :] = self.buffer[0:self.n-1, :, :]
        self.buffer[0, :, :] = im
        return self.buffer.copy(), reward, done, info
        # return self.buffer, reward, done, info

    def close(self):
        self.env.close()

def plot_frames(frames):
    n = frames.shape[0]
    fig, axs = plt.subplots(1, n)
    for i in range(n):
        axs[i].imshow(frames[i], cmap='Greys')
    plt.show()

def make_video(frames, savepath, bw=True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(frames.shape)
    writer = cv2.VideoWriter(str(savepath), fourcc, 25, (frames.shape[2], frames.shape[1]), isColor=not bw)
    for im in frames:
        writer.write(im)
    # print(im.shape)
    print('saving video to', savepath)
    writer.release()
    cv2.destroyAllWindows()

def interact_with_env(env, model_view=False, savepath=None):
    assert model_view and savepath or not model_view
    idx=0
    ims = []
    im = env.reset()
    # env.step(1)
    print(f"Enter number from 0 to {env.action_space.n-1} \nPress q or ctrl-C to quit")
    action = None
    prev_action = action
    try:
        while True:
            idx += 1
            action = input()
            if action == "":
                if action is not None:
                    action = prev_action
            if action == "q":
                break
            else:
                try:
                    action = int(action)
                    im_bw, rew, done, _ = env.step(action)
                    ims += [im_bw]
                    if done:
                        print("Resetting Env")
                        env.reset()
                    env.render()
                    prev_action = action
                except Exception as E:
                    import traceback
                    print(traceback.format_exc())
    except KeyboardInterrupt:
        if model_view:
            print('saving video (modelview) to: ', savepath)
            make_video(np.concatenate(ims, axis=0), savepath)
        env.close()

    if model_view:
        import ipdb; ipdb.set_trace()
        print('saving video (modelview) to: ', savepath)
        make_video(np.concatenate(ims, axis=0), savepath)
    env.close()

if __name__ == "__main__":
    import argparse
    import time
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="enter name of game env to load, I.e Breakout-v0, Boxing-v0", type=str)
    args = parser.parse_args()

    interactive = True

    if args.game is not None:
        game = args.game
    else:
        game = "Boxing-v0"

    imdir = Path(os.getcwd())/'images/'
    if not imdir.is_dir():
        os.mkdir(imdir)
    gamedir = imdir/game
    if not gamedir.is_dir():
        os.mkdir(gamedir)

    env = make_env(game)
    im = env.reset()

    print(env.observation_space.shape)
    print(env.action_space)
    # print(env.unwrapped.get_action_meanings())

    cv2.imwrite(str(gamedir/f"test.jpg"), env.frame)

    if interactive:
        interact_with_env(env, model_view=True, savepath=gamedir/f"interactive_model_view.mp4")
    else:
        print('writing test images to: ', gamedir)
        for i in range(10):
            idx += 1
            im_bw, _, _, _ = env.step((random.randint(0, env.action_space.n - 1)))
            cv2.imwrite(str(gamedir/f"test_{idx}.jpg"), env.frame)
