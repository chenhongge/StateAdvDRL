Robust Deep Reinforcement Learning against Adversarial Perturbations on Observations
--------------------

<p align="center">
<img src="http://www.huan-zhang.com/images/upload/state_adversarial/state_perturbation.png" width="50%" height="50%">
</p>

A robust deep reinforcement learning agent should be robust against observation
perturbations.  For example, a self-driving car can observe its location
through GPS, however GPS signal contains uncertainty and the driving policy
must take this uncertainty into consideration to plan routes safely.  To
guarantee safety under even the worst case uncertainty, we consider the
adversarial setting and study a modified MDP with adversary on state
observations (SA-MDP). Based on analysis of SA-MDP, we propose theoretically
principled regularizers for DQN and DDPG and borrow techniques from neural
network verification literature
([CROWN](https://github.com/huanzhang12/CROWN-Robustness-Certification) and
[CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP)) to train neural network
polices with robustness certificates.  More details of our algorithms can be
found in our paper:

*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Bo Li, Duane Boning,* and *Cho-Jui Hsieh*, "Robust Deep Reinforcement Learning against Adversarial Perturbations on Observations", [arxiv.org/abs/2003.08938](http://arxiv.org/abs/2003.08938) (\* Equal contribution)

Code 
-------------------------------------

In our paper, we conduct experiments on deep Q network (DQN) for discrete
action space environments (e.g. Atari games) and deep deterministic policy
gradient (DDPG) for continous action space environments. Our robust DQN and
DDPG algorithms (SA-DQN and SA-DDPG) only include one additional regularizer
during training.  Our reference implementation will be released in [DQN](DQN)
and [DDPG](DDPG) folders.
