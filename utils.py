import os
import psutil
import torch
import numpy as np
import cv2


def mem():
    """ Returns currently used memory in mb """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def buffer_memory_size(rb):
    """ Util to calculate how much memory is needed to allocate the full buffer, given the size of a last-filled entry"""
    idx = len(rb) - 1
    if idx > 0:
        dat = rb[idx]
        sizes = [sys.getsizeof(ele) for ele in (dat.state, dat.action, dat.reward, dat.next_state, dat.done)]
        ele_size = sum(sizes)
    else:
        return 0
    return len(rb) * ele_size

def make_video(frames, savepath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(savepath), fourcc, 25, (frames.shape[2], frames.shape[1]))
    for i in range(len(frames)):
        im = frames[i]
        writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    writer.release()
    cv2.destroyAllWindows()

def huber_loss(pred, target, k=1, reduction='mean'):
    assert reduction in ('mean', 'sum')
    td_err = torch.abs(pred-target)
    loss = torch.where(td_err < k, 0.5 * (td_err**2) / k, (n - 0.5*k))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def huber_quantile_loss(pred, target, tau=0.5, k=1, reduction='mean'):
    assert reduction in ('mean', 'sum')
    td_err = torch.abs(pred-target)
    huber_loss = torch.where(td_err < k, 0.5 * (td_err**2) / k, (n - 0.5*k))
    loss = torch.abs(tau - (td_err<0).float()) * huber_loss
    loss = loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert loss.shape == (pred.shape[0], 1)
    return loss
