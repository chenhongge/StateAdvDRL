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

*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Mingyan Liu, Bo Li, Duane Boning,*
and *Cho-Jui Hsieh*, "Robust Deep Reinforcement Learning against Adversarial
Perturbations on Observations", [**NeurIPS 2020
(Spotlight)**](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf)
(\*Equal contribution).  

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

Reference implementation for SA-DDPG can be found at [https://github.com/huanzhang12/SA_DDPG](https://github.com/huanzhang12/SA_DDPG).

## Pretrained agents performance

### Pretrained DQN agents 

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

### Pretrained PPO agents 

We repeatedly train each agent configuration at least 15 times, and rank them
with their average cumulative rewards over 50 episodes under the strongest
attack (among 5 attacks used). We report the performance for agents with
**median** robustnes (we do not cherry-pick the best agents).

| Environment | Evaluation                                 | Vanilla PPO | SA-PPO (convex) | SA-PPO (SGLD) |
|-------------|--------------------------------------------|-------------|-----------------|---------------|
| Humanoid-v2 | No attack                                  |   5270.6    |     6400.6      |     6624.0    |
|             | Strongest attack                           |   884.1     |     4690.3      |    **6073.8** |
| Walker2d-v2 | No attack                                  |   4619.5    |     4486.6      |    4911.8     |
|             | Strongest attack                           |   913.7     |     2076.1      |    **2468.4** |
| Hopper-v2   | No attack                                  |   3167.6    |     3704.1      |    3523.1     |
|             | Strongest attack                           |   733       |     1224.2      |    **1403.3** |

See our [SA-PPO repository](https://github.com/huanzhang12/SA_PPO) for more details.

### Pretrained DDPG agents

We attack each agent with 5 different attacks (random attack, critic attack,
MAD attack, RS attack and RS+MAD attack). Here we report the lowest reward of
all 5 attacks in "Strongest attack" rows. Additionally, we train each setting
11 times and we report the agent with median robustness (we do not cherry-pick
the best results). This is important due to the potential large training
variance in RL.


| Environment         | Evaluation       | Vanilla DDPG | SA-DDPG (SGLD) | SA-DDPG (convex) |
|---------------------|------------------|--------------|----------------|-----------------------------|
| Ant-v2              | No attack        | 1487         | 2186           | 2254                        |
|                     | Strongest attack | 142          | **2007**       | 1820                        |
| Walker2d-v2         | No attack        | 1870         | 3318           | 4540                        |
|                     | Strongest attack | 790          | 1210           | **1986**                    |
| Hopper-v2           | No attack        | 3302         | 3068           | 3128                        |
|                     | Strongest attack | 606          | **1609**       | 1202                        |
| Reacher-v2          | No attack        | -4.37        | -5.00          | -5.24                       |
|                     | Strongest attack | -27.87       | **-12.10**     | -12.44                      |
| InvertedPendulum-v2 | No attack        | 1000         | 1000           | 1000                        |
|                     | Strongest attack | 92           | 423            | **1000**                    |

See our [SA-DDPG repository](https://github.com/huanzhang12/SA_DDPG) for more details.

