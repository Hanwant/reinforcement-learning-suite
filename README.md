# Reinforcement Learning Suite

  This repo contains pytorch implementations of various RL algorithms, with a focus on Neural Networks as function approximators.
  Implementations are tested on various classic Atari games, as well as cartpole for quick prototyping. The main interface for running training and/or testing of agents is via run.py. <br>
  To run testing/training, execute run.py. <br>
  In addition to some command line parameters, there is a params dict inside the main function of run.py
  which has additional settings like type of model etc. 

  ```bash 
  python run.py --test Boxing-v0 --max_test_steps 1000 --use_cuda
  python run.py --train Boxing-v0 --continue_exp --max_steps 1000000 --max_test_steps 1000 --use_cuda
  python run.py --help for help
```

## Agents/Methods currently implemented:
     * DQN [1]
     * Double DQN [2]
     * Dueling DQN [3]
     * N-step DQN [7, 9]
     * Implicit Quantile Networks [4] (IQN)

   These agents typically share a common base (i.e DQN) and so are composable with each other (I.e As in Rainbow⁶)

  ![Results for an experiment run of an IQN agent on the Boxing-v0 Atari game](/examples/boxing_iqn_plot.png)

  <!-- Video of an IQN agent playing Breakout - after ~9,000,000 interactions with the environment -->
  ![Video of IQN playing Boxing](/examples/boxing_Iqn_step-14825908-reward-920.gif)
  ![Video of IQN playing Breakout](/examples/breakout_iqn_step-9900396-reward-72.gif)

## Agents/Methods to Implement:
   * Prioritized experience replay [8]
   * Fully Parameterized Quantile Function [5] (FQF)
  
## To Do:
   * Better config interface than hard-coding in script


### References
 1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529-533.
 2. Van Hasselt, H., Guez, A., & Silver, D. (2015). Deep reinforcement learning with double q-learning. arXiv preprint arXiv:1509.06461.
 3. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003).
 4. Dabney, W., Ostrovski, G., Silver, D., & Munos, R. (2018). Implicit quantile networks for distributional reinforcement learning. arXiv preprint arXiv:1806.06923.
 5. Yang, D., Zhao, L., Lin, Z., Qin, T., Bian, J., & Liu, T. Y. (2019). Fully parameterized quantile function for distributional reinforcement learning. In Advances in Neural Information Processing Systems (pp. 6193-6202).
 6. Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2017). Rainbow: Combining improvements in deep reinforcement learning. arXiv preprint arXiv:1710.02298.
 7. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937).
 8. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.
 9. Hernandez-Garcia, J. F., & Sutton, R. S. (2019). Understanding multi-step deep reinforcement learning: A systematic study of the DQN target. arXiv preprint arXiv:1901.07510.  
