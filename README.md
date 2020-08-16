# Reinforcement Learning Suite

  This repo contains pytorch implementations of various RL algorithms, with a focus on Neural Networks as function approximators.
  Implementations are tested on various classic Atari games, as well as cartpole for quick prototyping. The main interface for running training and/or testing of agents is via run.py. Execute python run.py --h to see how to pass arguments to the script. 

## Agents currently implemented:
     *  DQN¹
     *  Double DQN²
     *  Dueling DQN³
     *  Implicit Quantile Networks⁴ (IQN)
     * *Fully Parameterized Quantile Function⁵ (FQF)

   These agents typically share a common base (i.e DQN) and so are composable with each other (I.e As in Rainbow⁶)
     *Needs to be fixed

  ![Results for an experiment run of an IQN agent on the Boxing-v0 Atari game](/logs/Boxing-v0/16/plot.png)

  Video of an IQN agent playing Breakout - after ~9,000,000 interactions with the environment
  ![Video of IQN playing Breakout](https://github.com/Hanwant/reinforcement-learning-suite/blob/master/images/Breakout-v0/24/step_9900396_reward_72.0.mp4)
  <video src="/images/Breakout-v0/24/step_9900396_reward_72.0.mp4" width="320" height="200" controls preload></video>
## Agents to Implement:
   - N-step DQN [7, 14]
   - Prioritized experience replay [8]
   - Policy Gradient [9]
   - PPO [10]
   - DPG [11,12]
   - D4PG [13]
   - Actor-Critic Methods
  


### References
 1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529-533.
 2. Van Hasselt, H., Guez, A., & Silver, D. (2015). Deep reinforcement learning with double q-learning. arXiv preprint arXiv:1509.06461.
 3. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003).
 4. Dabney, W., Ostrovski, G., Silver, D., & Munos, R. (2018). Implicit quantile networks for distributional reinforcement learning. arXiv preprint arXiv:1806.06923.
 5. Yang, D., Zhao, L., Lin, Z., Qin, T., Bian, J., & Liu, T. Y. (2019). Fully parameterized quantile function for distributional reinforcement learning. In Advances in Neural Information Processing Systems (pp. 6193-6202).
 6. Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2017). Rainbow: Combining improvements in deep reinforcement learning. arXiv preprint arXiv:1710.02298.
 7. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937).
 8. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.
 9. Duan, Y., Chen, X., Houthooft, R., Schulman, J., & Abbeel, P. (2016, June). Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning (pp. 1329-1338).
 10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
 11. Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, June). Deterministic policy gradient algorithms.
 12. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
 13. Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W., Horgan, D., Tb, D., ... & Lillicrap, T. (2018). Distributed distributional deterministic policy gradients. arXiv preprint arXiv:1804.08617. 
 14. Hernandez-Garcia, J. F., & Sutton, R. S. (2019). Understanding multi-step deep reinforcement learning: A systematic study of the DQN target. arXiv preprint arXiv:1901.07510.  
