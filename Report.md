[image-tennis]: https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/generated_run_tennis.gif?raw=true "Trained Agent"
[rolling1]: https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/run_allrolling_without_ppo.png?raw=true "All rolling runs wihtout PPO"
[rolling2]: https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/run_allrolling_with_ppo.png?raw=true "All Rolling with PPPO"

# AI Agent 🤖 to Play a Tennis 🏸 match [**Report**]

------

![Trained Agent with DDPG for tennis][image-tennis]<br>
Image of player playing with DDPG


## Introduction
This project aimed to get a sense and discover the potential of 
algorithms such as Deep Deterministic Policy Gradients (DDPG) and 
Proximity Policy Organization (PPO) in a game 
competition between 2 agents in the field of Reinforced Learning (RL).

Initially this project aimed to use a basic DDPG 
(with pytorch) to solve the problem of an AI agent trying 
to learn how to play tennis.

After implementing sucessfully the algorithm, tried to explore the impact
of using PER. There was a visible advantage of using it.

Given that, tried to see the implication of using PPO, which has also interesting results.


## Methodology

I started by taking the notes from the [Udacity class - Reinforced Learning](https://classroom.udacity.com/nanodegrees/nd893/dashboard/overview)
and implementing a simple DDPG for the Tennis problem.
It worked very good, with and without PER.

After successfully running and passing the criteria, I tought on exploring the power of [PPO](https://arxiv.org/pdf/1707.06347.pdf).
Found online several implementations of the algorithm, some a bit more complex than others like 
[this example from Towards Data Science](https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc?gi=1e7667c5ac9d) 
or other [github example from kotsonis](https://github.com/kotsonis/ddpg-crawler) or [from ShangtongZhang](https://github.com/ShangtongZhang/DeepRL). <br>
Finaly I decided to readapt the an interesting [version from nikhilbarhate99](https://github.com/nikhilbarhate99/PPO-PyTorch)


## Learning Algorithm - Tennis

### DDPG Learning

The learning is done by the [Agent class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py),
together with the [Model class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ddpg.py) which represents the local and target 
Network. In order to ease the training, the agent uses a 
[ReplayBuffer class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py#L190)
as well as a [Ornstein-Uhlenbeck process class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py#L285).

### PPO Learning

The learning is done by the [Agent class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ppo.py),
together with the [Model class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ppo.py#) which represents the local and target 
Network. In order to ease the training, the agent uses a 
[RolloutBuffer class](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ppo.py#L11).


### Agent Details

The agent actions and learning are defined at [utils/agent_tennis_ddpg.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py).

The learning takes bellow initial parameters that guide the training:
```
Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of how many agents are training
            buffer_size (int): memory buffer size for ReplayBuffer
            batch_size (int): batch size for sampling to learn
            gamma (float): discount factor
            tau (float): interpolation parameter
            lr_actor (float): learning rate for Actor NN
            lr_critic (float): learning rate for Critic NN
            fc1_units (int): first hidden layer size
            fc2_units (int): second hidden layer size
            weight_decay (float): decay for Layer 2
            noise_mu (float): average factor for Ornstein-Uhlenbeck noise
            noise_theta (float): theta factor calculating Ornstein-Uhlenbeck noise
            noise_sigma (float): initial sigma factor to calculate Ornstein-Uhlenbeck noise
            noise_sigma_min (float): minimum sigma factor to calculate Ornstein-Uhlenbeck noise
            noise_sigma_decay (float): % to update sigma to calculate Ornstein-Uhlenbeck noise
        """
```

#### Agent functions:


**Step Function** to save each action and respecitve experience (rewards, ...) and learn from that step: <br>
`def step(self, state, action, reward, next_state, done):`

**Act Function** which takes a state and returns accordingly the action as per current policy  <br>
`def act(self, state, add_noise=True)`

**Learn Function** which updates accordingly the networks (Actor and Critic) <br>
`def learn(self, experiences)`

**Soft Update Function** performs a soft update to the model parameters <br>
`def soft_update(self, local_model, target_model, tau):`

**Reset Function** which resets the noise class
`def reset(self):`

#### Agent Auxiliary variables

Besides the functions described above, the Agent also uses a set of Variables/Objects to help its functioning.<br>
Out of which, important to mention **memory** which is an object of [ReplayBuffer](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py#L190) class
as well as **self.actor_local** and **self.critic_local** which is an object of [Actor and Critic](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ddpg.py).
Besides that, at each step there's also a [Noise element](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ddpg.py#L285) added to help the agent.

### Model Architecture DDPG

Used the same hidden sizes as the previous project.
Which means:
* **fc1_units:** 400
* **fc2_units:** 300

The NeuralNetwork for both Actor and Critic (defined at [model_tennis_ddpg.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ddpg.py)) 
are composed by 3 initial Linear Layers. (made them flexbile to receive the hidden sizes via parameter) <br>
The 1st and last Layer are the same for both Actor and Critic:

```
self.fc1 = nn.Linear(state_size, fc1_units)
(...)
self.fc3 = nn.Linear(fc2_units, action_size)
```

But the 2nd Linear Layer differs size, given that the Critic needs to **"critize"**
the choices made by the Actor. For that, the critic has embedded the actions in its hidden unit.

```
Actor: self.fc2 = nn.Linear(fc1_units, fc2_units)
Critic: self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
```
This will obviously mean that the forward method is also different, 
where in the case for the critic, the forward method gets the current state, and 
the actor's calculated actions, which are appended to the hidden layer. 

That results in the following Critic Forward:

```
def forward(self, state, action):
    state = state.to(device)
    action = action.to(device)
    """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
    xs = F.relu(self.fcs1(state))
    x = torch.cat((xs, action), dim=1)
    x = F.relu(self.fc2(x))
    return self.fc3(x)
```

Which differs from a more simple approach for the Actor Forward:

```
def forward(self, state):
    """Build an actor (policy) network that maps states -> actions."""
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return torch.tanh(self.fc3(x))
```

### Model Architecture PPO

The PPO model architecture is implemented at [model_tennis_ppo.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ppo.py))
which differs from DDPG where instead of seperate Classes for Actor and Critic,
we use a class for the [ActorCritic](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/model_tennis_ppo.py#L27).

I didn't change the network architecture, and went for the established to test it out.
Which seems to work, nevertheless, as I mention at [Ideas for the future](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md#ideas-for-the-future),
it would be very interesting to play with sizes of hidden layers and number of layers,
as well as different activations (as I will show bellow, here we are using Tahn as primary
activation, instead of ReLu, which come with its challenges).

Given that the problem/game at hand (Tennis) has a continuous action space,
then the Arquitecture is defined by a sequential model actor:
```
self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
```

which means 3 Linear layers (like the [DDPG Architecture](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md#model-architecture-ddpg))
and the 2 hidden layers both of size 64.

Instead of Rectified Linear Unit (ReLU) activation function in the hidden layer seen in  [DDPG Architecture](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md#model-architecture-ddpg))
it uses a Tanh activation function between the layers.
That might have an advantage, given that we are indeed using action which is spread between (-1, 1),
instead of (0, 1) like ReLu.
However it comes also with drawbacks, for example, the lenght of the training, given that we are increasing our universe
of numbers, we are getting a bigger sparsity.

For the critic, we also have a sequential model, defined as well by 3 layers,
which at the end, outputs an evaluation metric
like a "reward" on whether the action choosen by the actor was good or bad.

```
self.critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
    )
```

Like the actor architecture, it was choosen to use Tanh activation instead of the ReLu
in between the hidden layers.

The Policy of the Actor - Critic combination, gets updated every **xxx** ammount of timesteps
defined at the `train_ppo()` method in [Tennis.ipynb](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Tennis.ipynb) by the
variable `update_timestep`.<br>
The update is defined by the method [update() at utils/agent_tennis_ppo.py](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/utils/agent_tennis_ppo.py#L93)
And there we do an evaluation of the policies given the rewards, actions, and the log probabilities
iterating **xxx** times over the policy.


### ReplayBuffer (used in DDPG) 

Buffer used to store experiences which the Agent can learn from.<br>
The Buffer is initialized with the option to have **Prioritized Experience Replay** of not and adjusted the methods accordingly.

#### ReplayBuffer methods

**Add Function** to add an experience to the buffer <br>
`def add(self, state, action, reward, next_state, done)`

**Sample Function** that takes out a sample of `batch_size` from the buffer.
Depending on whether PER is on or not, it checks the priorities and weights accordingly before sampling.

`def sample(self, beta=0.03)`

**Update Priorities Function** given a set of indexes from the buffer, updates accordingly with the new priorities.
This function is only called in the case of PER (Prioritized Experience Replay) which updates the priorities / weights of the experiences accordingly after going through an experience.

`def update_priorities(self, indx, new_prio)`

### Hyper-Parameters
The project has some parameters which we can tweak to improve the performance. <br>
I didn't use the Optuna for tweaking the hyperparameters due to time constraints, but 
as suggested in the [Ideas for the future](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md#ideas-for-the-future)

#### DDPG Algorithm HyperParameters

For the DDPG Algorithm, one can have tweak following hyper-parameters:

* `'GAMMA'` # discount value used to discount each experiences
* `'TAU'` # value for interpolation
* `'LR_ACTOR'` # learning rate used for optimizer of gradient-descend
* `'LR_CRITIC'` # learning rate used for optimizer of gradient-descend
* `'FC1_UNITS'` # Values for 1st Hidden Linear Layer   
* `'FC2_UNITS'`# Values for 1st Hidden Linear Layer
* `'NOISE_SIGMA_MIN'` # lower limit of EPS (used for greedy approach)
* `'NOISE_SIGMA_DECAY'` # value for which EPS (used for greedy approach) is multiplied accordingly to decrease until reaching the lower limit

#### PPO Algorithm HyperParameters

* `'gamma'` # discount value used to discount each experiences
* `'tau'` # value for interpolation
* `'lr_actor'` # learning rate used for optimizer of gradient-descend
* `'lr_critic'` # learning rate used for optimizer of gradient-descend
* `'action_std'` # Starting Standard value for action distribution (Multivariate Normal)
* `'action_std_decay_rate'` # linearly decay rate for action distribution
* `'min_action_std'` # minimum value which action distribution can reach
* `'eps_clip'` # Clip parameter for PPO

*Note:* There are other parameters and dimensions that could be explored, but I thougth these could be the most relevant.  

### Individual Results

<table align="center">
        <tr><td>
<img src="https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/run_ddpg_without_per.png?raw=true" alt="Scores plot - DDPG without PER Run" align="center" height="300" width="350" >
<img src="https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/run_ddpg_with_per.png?raw=true" alt="Scores plot - DDPG with PER Run" align="center" height="300" width="350" >
<img src="https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Images/run_ppo.png?raw=true" alt="Scores plot - PPO Run" align="center" height="300" width="350" >

</table>
        

### Plotted rolling windows

![Rolling Averages without PPO][Rolling1]

![Rolling Averages with PPO][Rolling2]<br>


*Note:* Separeted with and without PPO, given that PPO requires a much larger ammount of 
episodes to reach the target so plot with PPO doesn't make it so easy to analyse 
DDPG with or without PER

## Conclusion

As you can see from the 
[charts](https://github.com/joao-d-oliveira/RL-TennisPlayers/blob/master/Report.md#plot-of-rewards) 
The DDPG PER is much faster to achieve the goal rather than without PER.

Another interesting and logical outcome, is that the smarter the agent, the slower 
it gets, given that the episode doesn't end so soon.

The PPO algorithm requires much more episodes to achieve the goal, however it isn't 
slower (or much slower than DDPG), the diference lays on the fact that the PPO is much more
steady and evolves gradually, and in the beg. when is not yet very "trained/intellegent"
it passes through the episodes rather fast. Unlike the DDPG which has more ups and downs
already from the beggining of the training. Therefore, in an initial DDPG training 
the score can be for example 1,07 or even higher ... which will lead to longer episodes.  

## Ideas for the Future

There is always room for improvement. <br>
From the vast number of ideas, or options that one can do to improve the performance of the agent, 
will try to least some of them to give some food-for-tought.

Some ideas for the future:

* Other algorithms could be tested as well, such as [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb), MADDPG
* Another possibility would be to spend more time finding the right HyperParameters;
* I really liked the performance and difference which PER makes in DDPG, I would like to try
with more time to try to add that advantage to the PPO algorithm;
* Another thing that one could try is to check other more complex (or less) models for the Actor and Critic, with fewer, or more layers, different activation functions, ...
* I did compare the overall time to reach the target, but unfortunately I didn't 
have a more timely measure episode by episode, this would be important to compare 
algorithms which have such a different number of episodes to train which (PPO vs DDPG vs MADDPG).
In the end what matters should be the evolution of the agents score over time, 
instead of over episodes;

------

## Rubric / Guiding Evaluation
[Original Rubric](https://review.udacity.com/#!/rubrics/1891/view)

#### Training Code

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Training Code  | The repository includes functional, well-documented, and organized code for training the agent. |
| ✅ Framework  | The code is written in PyTorch and Python 3. |
| ✅ Saved Model Weights | The submission includes the saved model weights of the successful agent. |

#### README

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ `README.md`  | The GitHub (or zip file) submission includes a `README.md` file in the root of the repository. |
| ✅ Project Details  | The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). |
| ✅ Getting Started | The README has instructions for installing dependencies or downloading needed files. |
| ✅ Instructions | The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. |


#### Report

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Report | The submission includes a file in the root of the GitHub repository or zip file 
(one of `Report.md`, `Report.ipynb`, or `Report.pdf`) that provides a description of the implementation. |
| ✅ Learning Algorithm | The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks. |
| ✅ Plot of Rewards | A plot of rewards per episode is included to illustrate that either: <br> * [version 1] the agent receives an average reward (over 100 episodes) of at least +30, or <br>* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.<br><br>The submission reports the number of episodes needed to solve the environment. |
| ✅ Ideas for Future Work | The submission has concrete future ideas for improving the agent's performance. |

#### Bonus :boom:
* ✅ Include a GIF and/or link to a YouTube video of your trained agent!

