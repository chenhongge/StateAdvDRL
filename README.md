# Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations

We study the robustness of deep reinforcement learning agents when their state
observations (e.g., measurements of environment states) contain noises or even
adversarial perturbations.  For example, a self-driving car can observe its
location through GPS, however GPS signal contains uncertainty and the driving
policy must take this uncertainty into consideration to plan routes safely.  To
guarantee performance under even the worst case uncertainty, we study the
foundamental state-adversarial Markov Decision Process (**SA-MDP**) and propose
theoretically principled robustness regularizers for **PPO**, **DDPG** and
**DQN**. In addition, we also propose two strong adversarial attacks for PPO
and DDPG, the maximal action difference (MAD) attack and the **robust sarsa
(RS) attack**. More details can be found in our paper:

*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Mingyan Liu, Bo Li, Duane Boning,* and *Cho-Jui Hsieh*, "Robust Deep Reinforcement Learning against Adversarial Perturbations on Observations", [arxiv.org/abs/2003.08938](http://arxiv.org/abs/2003.08938) (\* Equal contribution)

## Robust Deep Reinforcement Learning Demos (SA-DQN, SA-DDPG and SA-PPO)

| ![Pong-attack-natural.gif](/assets/Pong-attack-natural.gif) | ![Pong-attack-natural.gif](/assets/Pong-attack-convex.gif) | ![RoadRunner-attack-natural.gif](/assets/RoadRunner-attack-natural.gif) | ![RoadRunner-attack-natural.gif](/assets/RoadRunner-attack-convex.gif) | 
|:--:| :--:| :--:| :--:| 
| **Pong**, *Vanilla DQN* <br> reward under attack: **-21** <br> (trained agent: right paddle) | **Pong**, *SA-DQN* <br> reward under attack: **21** <br> (trained agent: right paddle) |**RoadRunner**, *Vanilla DQN* <br> reward under attack: **0** |**RoadRunner**, *SA-DQN* <br> reward under attack: **49900** |
| ![humanoid_vanilla_ppo_attack_615.gif](/assets/humanoid_vanilla_ppo_attack_615.gif) | ![humanoid_sappo_attack_6042.gif](/assets/humanoid_sappo_attack_6042.gif) | ![ant_vanilla_ddpg_attack_189.gif](/assets/ant_vanilla_ddpg_attack_189.gif) | ![ant_saddpg_attack_2025.gif](/assets/ant_saddpg_attack_2025.gif) |
| **Humanoid** *Vanilla PPO* <br> reward under attack: **615** | **Humanoid** *SA-PPO* <br> reward under attack: **6042** | **Ant** *Vanilla DDPG* <br> reward under attack: **189**  | **Ant** *SA-DDPG* <br> reward under attack: **2025** |

## Code 

In our paper, we conduct experiments on deep Q network (DQN) for discrete
action space environments (e.g. Atari games), deep deterministic policy
gradient (DDPG) for continous action space environments with deterministic
actions, and proximal policy optimization (PPO) for continous action space
environments with stochastic actions.  Our proposed algorithms (SA-PPO, SA-DDPG
and SA-PPO) are evaluated using 11 environments.

Reference implementation for SA-DQN can be found at [https://github.com/chenhongge/SA_DQN](https://github.com/chenhongge/SA_DQN).

Reference implementation for SA-PPO can be found at [https://github.com/huanzhang12/SA_PPO](https://github.com/huanzhang12/SA_PPO).

We are finalizing the reference implementation for SA-DDPG and it will be released soon.

## Pretrained model performance

### Pretrained DQN models 

| Environment | Evaluation         | Vanilla DQN | SA-DQN (convex relaxation) |
|-------------|--------------------|:-----------:|:--------------------------:|
| Pong        | No attack          |   21.0±0.0  |         21.0±0.0           |
|             | PGD 10-step attack |  -21.0±0.0  |       **21.0±0.0**         |
|             | PGD 50-step attack |  -21.0±0.0  |       **21.0±0.0**         |
| RoadRunner  | No attack          |  45534.0±7066.0  |   44638.0±7367.0      |
|             | PGD 10-step attack |   0.0±0.0   |     **44732.0±8059.5**     |
|             | PGD 50-step attack |   0.0±0.0   |     **44678.0±6954.0**     |
| BankHeist   | No attack          |  1308.4±24.1|      1235.4±9.8            |
|             | PGD 10-step attack |   56.4±21.2 |       **1232.4±16.2**      | 
|             | PGD 50-step attack |   31.0±32.6 |       **1234.6±16.6**      |
| Freeway     | No attack          |   34.0±0.2  |         30.0±0.0           |
|             | PGD 10-step attack |   0.0±0.0   |        **30.0±0.0**        |
|             | PGD 50-step attack |   0.0±0.0   |        **30.0±0.0**        |

See our [SA-DQN repository](https://github.com/chenhongge/SA_DQN) for more details.

### Pretrained PPO models 

| Environment | Evaluation                                 | Vanilla PPO | SA-PPO (convex) | SA-PPO (SGLD) |
|-------------|--------------------------------------------|-------------|-----------------|---------------|
| Walker2d-v2 | No attack                                  | 3357        | 3552            | 3917          |
|             | No attack (deterministic action)           | 3081        | **4939**        | 4617          |
|             | Robust Sarsa attack                        | 571         | 2496            | 1733          |
|             | Robust Sarsa attack (deterministic action) | 550         | **4700**        | 1999          |
| Hopper-v2   | No attack                                  | 2576        | 2261            | 2436          |
|             | No attack (deterministic action)           | 3574        | 3524            | **3698**      |
|             | Robust Sarsa attack                        | 635         | 1066            | 1086          |
|             | Robust Sarsa attack (deterministic action) | 617         | **1463**        | 1044          |
| Humanoid-v2 | No attack                                  | 2269        | 5067            | 5392          |
|             | No attack (deterministic action)           | 2008        | 6339            | **6760**      |
|             | Robust Sarsa attack                        | 637         | 4251            | 3848          |
|             | Robust Sarsa attack (deterministic action) | 567         | **5954**        | 5690          |


In our paper, we reported the performance on non-deterministic actions (where
we still sample from Gaussian distributions during evaluation). However, a more
appropriate evaluation procedure is to use deterministic action without noise
during evaluation, which improves performance significantly. **Thus the results
reported in our paper (Table 1) are too pessimistic,** and please refer to the
**deterministic action** rows in the table above for more accurate results
under the correct evaluation protocol (we will update numbers on our paper in a
later revision).

See our [SA-PPO repository](https://github.com/huanzhang12/SA_PPO) for more details.

### Pretrained DDPG models

(will be released soon)

