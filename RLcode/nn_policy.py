import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as torch_dist
import torch.nn.functional as F
import torch.optim as optim

def nn_policy(states):
    """
    a fully connected network which takes the observation as input 
    and produces the action a
    """
    return states



class Policy(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.affine1 = nn.Linear(state_dim, 100)
        self.affine2 = nn.Linear(100, 100)
        self.affine3 = nn.Linear(100, act_dim)

        self.activation = nn.Tanh()
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.activation(self.affine3(x))
        return x
        
    def select_action(self):
        x = torch.from_numpy(x).float().unsqueeze(0)
        mu = self.forward(x)
        sample_distr = Sample_Distribution.gaussian_policy(mu)

    def save(self, state_file='models/policy_network.pt'):
        # Save the model state
        torch.save(self.state_dict(), state_file)

    @staticmethod
    def load(state_file='models/policy_network.pt'):
        
        # Create a network object with the constructor parameters
        policy = Policy()
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
