import numpy as np
import random
from models import Actor, Critic
from ou_noise import OUNoise
from buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, hyper_params, PR=False, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        # priority replay buffer
        self.PR = PR
        self.step_cnt = 0
        self.learn_cnt = 0

        if hyper_params != None:
            self.buffer_size, self.batch_size, self.gamma, self.tau, self.lr_actor, self.lr_critic, self.weight_decay, self.fc1_units_actor, self.fc2_units_actor, self.fc1_units_critic, self.fc2_units_critic, self.fc3_units_critic, self.mu, self.theta, self.sigma = hyper_params
        else:
            self.buffer_size = int(909973)
            self.batch_size = 384
            self.gamma = 0.9901736020389454
            self.tau = 0.0018092674864975023
            self.lr_actor =  0.00017377360292847218
            self.lr_critic = 0.00019323989991911093
            self.weight_decay = 0.0007434075695379568   
            self.fc1_units_actor = 400
            self.fc2_units_actor = 300
            self.fc1_units_critic = 256
            self.fc2_units_critic = 256
            self.fc3_units_critic = 128
            self.theta = 0.20215485858989818
            self.sigma = 0.2053184766109179
            self.mu = -0.0036382192060112045

        self.n_experiences = 16
        self.n_learn = 16
        self.n_critic = 1

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, self.fc1_units_actor, self.fc2_units_actor).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, self.fc1_units_actor, self.fc2_units_actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, self.fc1_units_critic, self.fc2_units_critic).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, self.fc1_units_critic, self.fc2_units_critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # let target start with same values
        self.soft_update(self.critic_local, self.critic_target, 1.)
        self.soft_update(self.actor_local, self.actor_target, 1.)
        
        # Noise process
        self.noise = OUNoise(action_size, self.mu, self.theta, self.sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed, PR=self.PR)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        self.memory.add_multiple(states, actions, rewards, next_states, dones)
    
        self.step_cnt = (self.step_cnt + 1) % self.n_experiences
        if self.step_cnt == 0:
            if len(self.memory) >= self.batch_size:
                for _ in range(self.n_learn):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            if add_noise:
                action += self.noise.sample()

        self.actor_local.train()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.PR:
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.PR:
            # Compute TD-errors for PER update
            td_errors = Q_targets - Q_expected
            # Update priorities in replay buffer based on TD-errors
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())  

        # Calculate the squared errors
        if self.PR:
            #squared_errors = (Q_expected - Q_targets) ** 2
            #weighted_squared_errors = weights * squared_errors
            #critic_loss = weighted_squared_errors.mean()
            critic_loss = (weights * F.mse_loss(Q_expected, Q_targets)).mean()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
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
        self.learn_cnt = (self.learn_cnt + 1) % self.n_critic
        if True:#self.learn_cnt == 0:
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

