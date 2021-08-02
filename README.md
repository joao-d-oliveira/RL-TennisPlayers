[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# AI Agent 🤖 to Play a Tennis 🏸 match

## Introduction

This project aims to explore the power of teaching an agent 
through Reinforced Learning (RL) to learn how to play Tennis.  

In order to teach, we explored the options of teaching the agent through a Deep Deterministic Policy Gradients (DDPG) algorithm 
with Prioritized Experience Replay (PER), as well as with an Proximal Policy Optimization (PPO), to learn how to control efficiently the arm.

We are working with the  [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

## Project Details

### Environment

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

### Space State

The observation space consists of 8 variables 
corresponding to the position and velocity of the 
ball and racket. 
Each agent receives its own, local observation.  
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

So for example:

```
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```

### Conditions to consider solved

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Distributed Training

As mentioned above, we used the DDPG algorithm with and without PER,
 and noticed that PER makes a considerable difference.
(see [Report.md](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md) for further info)

To test the other options of algorithm, we tested the [PPO](https://arxiv.org/pdf/1707.06347.pdf)
algorithm which also delivers good results, however needs more episodes.

## Instructions

### Getting Started

Before starting any of the environments you need to install 
`unityagents` package from Unity (version 0.4.0).
As this is an old version, the best is to download the
[python folder](https://github.com/joao-d-oliveira/RL-TennisPlayers/tree/master/python) and run 
`cd python; pip install  .` to install it manually.

#### For Tennis Environment 

1. You need to have installed the requirements (specially mlagents==0.4.0).
   Due to deprecated libraries, I've included a [python folder](https://github.com/joao-d-oliveira/RL-TennisPlayers/tree/master/python) which will help
   with installation of the system.
      - Clone the repository: `git clone https://github.com/joao-d-oliveira/RL-TennisPlayer.git`
      - Go to python folder: `cd RL-TennisPlayer/python`
      - Compile and install needed libraries `pip install .`
2. Download the environment from one of the links below
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
3. Place the downloaded file for your environment in the DRLND GitHub repository, in the ``RL-TennisPlayer`` folder, and unzip (or decompress) the file.

### Action Space

Two continuous actions are available, corresponding to movement toward 
(or away from) the net, and jumping.

### Files

#### Code
1. [utils/agent_tennis_ddpg.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py) - Agent class containing methods to help the agent learn and acquire knowledge using DDPG algorithm
2. [utils/model_tennis_ddpg.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ddpg.py) - DDQG model of Actor and Critic class setup 
3. [utils/agent_tennis_ppo.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ppo.py) - Agent class containing methods to help the agent learn and acquire knowledge using PPO algorithm
4. [utils/model_tennis_ppo.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ppo.py) - PPO model of Actor and Critic class setup 
5. [Tennis.ipynb](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Tennis.ipynb) - Jupyter Notebook for running experiment, for Tennis Players

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md) - Detailed Report on the project

#### Models
All models are saved on the subfolder ([saved_models](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/saved_models)).
For example, [checkpoint_actor_ddpg_True-PER_1.pth](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/saved_models/checkpoint_actor_ddpg_True-PER_1.pth) 
and [checkpoint_critic_ddpg_True-PER_1.pth](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/saved_models/checkpoint_critic_ddpg_True-PER_1.pth) are 
files which has been saved upon success of achieving the goal, and
[finished_actor_ddpg_False-PER_1.pth](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/saved_models/finished_actor_ddpg_False-PER_1.pth) 
and [finished_critic_ddpg_False-PER_1.pth](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/saved_models/finished_critic_ddpg_False-PER_1.pth) 
are the end model after runing all episodes.

### Running Tennis training


#### Structure of Notebook

The structure of the notebook follows the following:
> 1. Collaboration and Competition <br>
> 1.1 Start the Environment: _(load environment for the game)_<br>
> 1.2 DDPG _(section to train agent with DDPG algorithm, with or without PER)_<br>
> 1.3 PPO: _(section to train agent with PPO algorithm)_<br>
> 2.0 Plot all results: _(section where all the results from above sections are plotted to compare performance)_

#### Running

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-TennisPlayers#getting-started) 
0. Load Jupyter notebook [Tennis.ipynb](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Tennis.ipynb)
1. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions
2. Choose which algorithm you want to train your agent with, PPO or DDPG
-----
