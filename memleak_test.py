import os
import psutil
import numpy as np
import torch
from Agent import ReplayBuffer, SARSD

DEVICE = 'cuda' 

def mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)

def make_data():
    states = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    actions = np.random.randint(0, 10, 1)
    rewards = np.random.normal(0, 1, 1)
    next_states = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    done = np.random.binomial(0.5, 1)
    return SARSD(states, actions, rewards, next_states, done)

def make_buffer(size):
    rb = ReplayBuffer(size)
    for i in range(size):
        rb.insert(make_data())
    return rb

def make_tensors(data):
    states = torch.tensor(data.state, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(data.action, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(data.reward, dtype=torch.float32, device=DEVICE)
    next_states = torch.tensor(data.next_state, dtype=torch.float32, device=DEVICE)
    done  = torch.tensor(data.done, dtype=torch.bool, device=DEVICE)
    loss = (rewards + (states + next_states).mean() * done).mean()
    return loss.detach().item()

def loop1():
    buff_size = 10000
    rb = ReplayBuffer(buff_size)
    try:
        for i in range(buff_size):
            data = make_data()
            rb.insert(data)
            if i % 1000 == 0:
                print(i)
                print(len(rb))
                print("memory (mb) : ", mem())

        for i in range(buff_size):
            data = rb.sample(32)
            mem_base = mem()
            loss = make_tensors(data)
            mem1 = mem()
            if i % 1000 == 0:
                print(i)
                print("memory_accumulated: ", mem1 - mem_base)
    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()
    except:
        raise

if __name__=="__main__":

    rb = make_buffer(1000)
    mem_base = mem()
    for i in range(10):
        # data = rb.sample(32)
        data = np.random.choice(rb.buffer, 32)
        # data = make_data()
        # state = torch.tensor(data.state, dtype=torch.float32)#.to(DEVICE)
        # print(state.shape, state.dtype)
        print(mem() - mem_base)








