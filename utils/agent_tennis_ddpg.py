import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.model_tennis_ddpg import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDDPG:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, num_agents,
                 buffer_size=int(1e5),
                 batch_size=128,
                 gamma=0.99,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 fc1_units=400,
                 fc2_units=300,
                 weight_decay=0,  # L2 weight decay
                 noise_mu=0.,
                 noise_theta=0.15,
                 noise_sigma=0.2,
                 noise_sigma_min=1e-4,
                 noise_sigma_decay=0.97,
                 prioritized_experience_learning=False
                 ):
        """Initialize an Agent object.

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
        self.state_size = state_size
        self.action_size = action_size
        self.PER = prioritized_experience_learning

        random.seed(random_seed)

        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(
            device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(
            device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(
            device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(
            device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed, noise_mu, noise_theta, noise_sigma,
                             noise_sigma_min, noise_sigma_decay)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, prioritized_experience_replay=prioritized_experience_learning)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # old
        # self.memory.add(state, action, reward, next_state, done)
        # added to save states agent by agent
        for i in range(state.shape[0]):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = np.array([self.actor_local(s).cpu().data.numpy() for s in state])

        self.actor_local.train()
        if add_noise: actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ?? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.PER: states, actions, rewards, next_states, dones, idx, weights = experiences
        else: states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.PER:
            weights = torch.from_numpy(weights)
            weights = weights.unsqueeze(1)
            Q_targets *= weights
            Q_expected *= weights

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        if self.PER:
            # update older priorities with new losses distances
            new_priorities = abs(Q_expected - Q_targets)
            self.memory.update_priorities(idx, new_priorities)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_experience_replay=False, alpha=0.05):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_experience_replay(bool): whether to have Prioritized Experience Replay (PER). or not
            alpha: value to use in (PER) calculation of probabilities
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.PER = prioritized_experience_replay
        self.alpha = alpha
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        if self.PER:
            self.max_priority = 0
            self.priorities = deque(maxlen=buffer_size)
        #     self.buffer_pos = 0
        #     self.prioritized_experience = np.empty(buffer_size, dtype=[("priority", np.float32), ("experience", self.experience)])
        #     self.priorities = np.ones((buffer_size,), dtype=np.float32)

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        n = state.shape[0] if len(state.shape) > 1 else 1

        e = self.experience(state, action, reward, next_state, done)
        if self.PER:
            prio = 1 if len(self.priorities) < self.batch_size else max(self.priorities)
            self.priorities.append(prio)

        self.memory.append(e)

        # for i in range(n):
        #     if len(state.shape) > 1: e = self.experience(state[i], action[i], reward[i], next_state[i], done[i])
        #     else: e = self.experience(state, action, reward, next_state, done)
        #     if self.PER:
        #         prio = 1 if len(self.priorities) < self.batch_size else max(self.priorities)
        #         self.priorities.append(prio)
        #
        #     self.memory.append(e)


    def sample(self, beta=0.03):
        """Randomly sample a batch of experiences from memory."""
        if not self.PER:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            # X is the number of steps before updating memory
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()

            length = len(self.memory)
            idxs = np.random.choice(np.arange(length), size=self.batch_size, replace=True, p=probs)
            weights = (length * probs[idxs]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            # select the experiences and compute sampling weights
            experiences = [self.memory[i] for i in idxs]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        if not self.PER: return states, actions, rewards, next_states, dones
        else: return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, indx, new_prio):
        """Updates priority of experience after learning."""
        for new_p, i in zip(new_prio, indx):
            new_p = new_p.item() ** self.alpha
            self.priorities[int(i)] = new_p
            if new_p > self.max_priority: self.max_priority = new_p

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, sigma_min=1e-4, sigma_decay=0.97):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.state = copy.copy(self.mu)
        self.theta = theta
        self.sigma = sigma
        self.min_sigma = sigma_min
        self.decay_sigma = sigma_decay
        random.seed(seed)
        self.reset()

    def decrease_sigma(self):
        self.sigma = max(self.min_sigma, self.sigma * self.decay_sigma)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.decrease_sigma()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
