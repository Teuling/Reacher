import random
from collections import deque, namedtuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, PR=False, alpha=0.6, beta=0.4, epsilon = 1e-6):
        """Initialize a ReplayBuffer object. """
        self.PR = PR
        self.alpha = alpha
        self.beta = beta
        self.action_size = action_size
        self.epsilon = epsilon
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        if self.PR:
            self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if self.PR:
            # Assign max priority to new experience
            max_priority = max(self.priorities, default=1.0)
            self.priorities.append(max_priority)

    def add_multiple(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.add(state, action, reward, next_state, done) 
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.PR:
            # Calculate sampling probabilities
            scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
            sampling_probs = scaled_priorities / sum(scaled_priorities)
            # Sample indices based on sampling probabilities
            indices = np.random.choice(len(self.memory), size=self.batch_size, p=sampling_probs)
            experiences = [self.memory[idx] for idx in indices]
            # Importance-sampling weights to correct for bias
            weights = (len(self.memory) * sampling_probs[indices]) ** -self.beta
            # Normalize weights
            weights /= weights.max()
            weights = torch.from_numpy(np.vstack([e for e in weights if e is not None]).astype(np.uint8)).float().to(device)
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if self.PR:
            return states, actions, rewards, next_states, dones, weights, indices
        else:
            return states, actions, rewards, next_states, dones
    
    def update_priorities(self, indices, td_errors):
        for idx, td_erro in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_erro) + self.epsilon) ** self.alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)