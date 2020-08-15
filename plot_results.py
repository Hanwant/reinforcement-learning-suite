import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import click

from run import get_latest_id


@click.command()
@click.argument('game', default = 'CartPole-v0')
@click.option('--basepath', default = None, help='parent directory to search for logs - assumes saved as per run.py')
@click.option('--exp_id', default = None, help='If not given, assumes most recent experiment')
@click.option('--save/--nsave', default = False, help='Save to png')
@click.option('--savepath', default = None, help='path to save png. If not given, saves in same folder as log')
def main(game, basepath, exp_id, save, savepath):
    """
    Plots logs for loss and rewards for an experiment run by executing run.py "game" --train
    """
    if basepath is None:
        basepath = Path(os.getcwd())
    if exp_id is None:
        exp_id = get_latest_id(basepath, game)
    if save:
        if savepath is None:
            savepath = basepath/f'logs/{game}/{exp_id}/plot.png'
    print('exp_id: ', exp_id)
    logpath = basepath/f'logs/{game}/{exp_id}/testlog.csv'
    logs = pd.read_csv(logpath, sep=',', header=0)
    rolling_period = 10
    fig, ax = plt.subplots(2)
    fig.suptitle(f'Environment: {game} Experiment: {exp_id}')
    ax[0].plot(logs['total_steps'], logs['loss'], label="raw")
    ax[0].plot(logs['total_steps'], logs['loss'].rolling(rolling_period, min_periods=1).mean(), label=f"{rolling_period} period rolling mean")
    ax[0].set_xlabel('total environment steps')
    ax[0].set_ylabel('loss')
    ax[0].legend(fontsize=8)
    ax[1].plot(logs['total_steps'], logs['reward'], label="raw")
    ax[1].plot(logs['total_steps'], logs['reward'].rolling(rolling_period, min_periods=1).mean(), label=f"{rolling_period} period rolling mean")
    ax[1].set_xlabel('total environment steps')
    ax[1].set_ylabel('reward')
    ax[1].legend(fontsize=8)
    plt.show()
    if save:
        print('saving plot to:', savepath)
        fig.savefig(Path(savepath))


if __name__ == "__main__":
    main()
