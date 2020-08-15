RL_SUITE

This repo contains implementations in pytorch of various RL algorithms, with a focus on Neural Networks as functions approximators.
Implementations are tested on various classic Atari games, as well as cartpole for quick prototyping. The main interface for running training and/or testing of agents is via run.py. Execute python run.py --h to see how to pass arguments to the script. 

Agents currently implemented:
DQN¹
Double DQN²
Dueling DQN³
Implicit Quantile Networks⁴ (IQN)
Fully Parameterized Quantile Function⁵ (FQF)

To do list:
N-step DQN⁶
Prioritized experience replay⁷

These agents are all based on a base DQN agent and are to be composable with each other (I.e Rainbow⁸)


References
1. 

8. 
