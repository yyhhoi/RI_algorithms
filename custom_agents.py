import torch
import numpy as np
class AgentGrad():
    def __init__(self, obs_dim, act_dim, epsilon=0.2, policy='epsilon-greedy'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epsilon = epsilon
        self.policy = policy



    def a_epsilon_greedy(self, q_alla):
        if np.random.rand() < self.epsilon:
            out_a = int(np.random.choice(self.act_dim))
        else:
            out_a = torch.argmax(q_alla).item()
        return out_a
    
    def select_action(self, s):
        if self.policy == 'epsilon-greedy':
            return self.a_epsilon_greedy(s)