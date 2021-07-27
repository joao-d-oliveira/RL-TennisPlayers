[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# AI Agent 🤖 to Play a Tennis 🏸 match

## Introduction

This project aims to explore the power of teaching an agent 
through Reinforced Learning (RL) to learn how to play Tennis.  

**TODO: CHANGE**: For this we are using a Deep Deterministic Policy Gradients (DDPG) algorithm 
to learn how to control efficiently the arm.

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

So for example, in the case of Multi Agent version (2º version):

-----
**TODO CHANGE:**

```
There are 20 agents. Each observes a state with length: 33
The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
```
-----

### Conditions to consider solved

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Distributed Training

-----
**TODO CHANGE:**
For this project, we can either use:
- The first version of the environment with a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

For this we used the DDPG algorithm for both the 1st and 2nd version, so with 1 or 20 agents, and noticed that with the Multi version the results are much faster
(see [Report.md](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md) for further info)

With the optional challenge [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler), 
we see that DDPG is not enough (or too slow) to converge, so for this I tried to search for another algorithm such 
as [PPO](https://arxiv.org/pdf/1707.06347.pdf).
-----

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

#### For Soccer Environment

![Soccer][image2]

In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)
You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `RL-TennisPlayer/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Soccer.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents.  (_To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


### Action Space

-----
**TODO CHANGE:**
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
-----

### Files

-----
**TODO CHANGE:**

#### Code
1. [utils/agent_reacher_ddpg.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py) - Agent class containing methods to help the agent learn and acquire knowledge using DDPG algorithm
1. [utils/model_reacher.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/model_reacher.py) - DDQG model of Actor and Critic class setup 
1. [Continuous_Control.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Continuous_Control.ipynb) - Jupyter Notebook for running experiment, for Reacher Control
---
1. [utils/agent_crawler_ppo.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_crawler_ppo.py) - Agent class containing methods to help the agent learn and acquire knowledge using PPO algorithm for Crawler environment
1. [utils/model_crawler.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/model_crawler.py) - PPO model of Actor and Critic class setup 
1. [Crawler.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Crawler.ipynb) - Jupyter Notebook for running experiment, for Crawler Control

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Report.md) - Detailed Report on the project

#### Models
All models are saved on the subfolder ([saved_models](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/saved_models)).
For example, [checkpoint_Mult_actor.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/saved_models/checkpoint_Mult_actor.pth) 
and [checkpoint_Mult_critic.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/saved_models/checkpoint_Mult_critic.pth) are 
files which has been saved upon success of achieving the goal, and
[finished_Reacher_Mult_actor.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/saved_models/finished_Reacher_Mult_actor.pth) 
and [finished_Reacher_Mult_critic.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/saved_models/finished_Reacher_Mult_critic.pth) 
are the end model after runing all episodes.
-----

### Running Tennis training

-----
**TODO CHANGE:**

#### Structure of Notebook

The structure of the notebook follows the following:
> 1. Initial Setup: _(setup for parameters of experience, check [report](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Report.md) for more details)_ <br>
> 2. Continuous Control <br>
> 2.1 Start the Environment: _(load environment for the game)_<br>
> 2.2 Helper Functions: _(functions to help the experience, such as Optuna, DDPG, ...)_<br>
> 2.3 Baseline: _(section to train an agent with the standard parameters, without searching for hyper-parameters)_<br>
> 2.4 HyperParameters: _(section to train an agent with hyperparameters)_<br>
> 3.0 Plot all results: _(section where all the results from above sections are plotted to compare performance)_

At Initial Setup, you can define whether you want 
One or Multiple agents by turning at `SETUP` dictionary:
- `'MULTI_ONE': 'One',  # 'Mult' or 'One'` 

#### Running

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-RobotArm#getting-started) 
0. Load Jupyter notebook [Continuous_Control.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Continuous_Control.ipynb)
1. Adapt dictionary `SETUP = {` with the desired paramenters
2. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions

-----

### Running Soccer Environment

-----
**TODO CHANGE:**

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-RobotArm#getting-started) 
0. Load Jupyter notebook [Crawler.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Crawler.ipynb)
1. Run all cells, no special options needed

-----