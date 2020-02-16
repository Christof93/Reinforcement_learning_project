import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as torch_dist
import torch.nn.functional as F
import torch.optim as optim

def dummy_feature(state, *_):
    return state

def nn_policy_func(**hyperparam_dict):
    """
    combines the feature vector linearly with weights to produce the action
    """

    nnPolicy = NN_Policy()
    nnPolicy.set_params(hyperparam_dict)
    nnPolicy.train()

    return nnPolicy

class NN_Policy(nn.Module):
    def __init__(self):
        super(NN_Policy, self).__init__()

    def set_params(self, hyperparam_dict):
        self.gamma = hyperparam_dict['gamma']
        self.learning_rate = hyperparam_dict['learning_rate']
        self.poly_degree = hyperparam_dict['poly_degree']
        self.state_size = hyperparam_dict['state_size']
        #self.feature_size = (self.poly_degree+1)**self.state_size
        self.activation_func = hyperparam_dict['activation_function']
        self.sample_func = hyperparam_dict['sample_function']
        self.sample_std = hyperparam_dict['sample_log_std']
        self.num_actions = hyperparam_dict['num_actions']

        self.affine1 = nn.Linear(self.state_size, 100)
        self.affine2 = nn.Linear(100, 100)
        self.affine3 = nn.Linear(100, self.num_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.activation_func(self.affine3(x))
        return x

    def select_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        mu = self.forward(x)
        mu, action, log_prob = self.sample_func(mu)

        # Also save the log of the probability for the selected action
        self.saved_log_probs.append(log_prob)

        # Return the chosen action value
        return action

    def save(self, state_file='models/nn_policy.pt'):
        # Save the model state
        torch.save(self.state_dict(), state_file)

    @staticmethod
    def load(hyperparams, state_file='models/nn_policy.pt'):
        
        # Create a network object with the constructor parameters
        policy = NN_Policy()
        policy.set_params(hyperparams)
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
