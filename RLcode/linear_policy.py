import random
import numpy as np
import math
import torch
import torch.nn as nn

from sample_distribution import *
from itertools import product
from functools import reduce


def poly_feature_slow(states, degree=2):
    '''
    turns the observation of the state into a polynomial feature 
    vector of degree n.
    Computes the polynomial feature vector from the input states
    
    :param: state: Input state vector of size k (1D numpy array)
    :param: n: polynomial degree 
    :return: feature vector phi consisting of (n+1)^k elements (1D numpy array)
    '''
    k = states.shape[0]
    phi = np.zeros((degree+1)**k)
    for i, _ in enumerate(phi):
        # convert num into base (n+1)
        num = i
        poly_degrees = list()
        for _ in range(k):
            p = num%(degree+1)
            poly_degrees.append(p)
            num = math.floor(num/(degree+1))        
        
        poly_degrees.reverse()
        calculated_degrees = np.array([(s**p) for s, p in zip(states, poly_degrees)])
        
        phi[i] = np.prod(calculated_degrees)
        
    return phi

def poly_feature(states, degree=2):
    k = states.shape[0]
    feature_size = (degree+1)**k
    n_ary_cart = []
    for s in states:
        poly = []
        for d in range(degree+1):
            poly.append(s**d)
        n_ary_cart.append(poly)
    phi = np.array([reduce(lambda x,y: x*y, prod) for prod in product(*n_ary_cart)])
    #assert feature_size == len(phi)
    return phi
        

def linear_fa_policy(**hyperparam_dict):
    """
    combines the feature vector linearly with weights to produce the action
    """
    

    linear_fa_policy = LinearFAPolicy()
    linear_fa_policy.set_params(hyperparam_dict)
    linear_fa_policy.train()
    
    return linear_fa_policy

class LinearFAPolicy(nn.Module):
    def __init__(self):
        super(LinearFAPolicy, self).__init__()
        # Degree of the polynomial feature
        
    def set_params(self, hyperparam_dict):
        self.gamma = hyperparam_dict['gamma']
        self.learning_rate = hyperparam_dict['learning_rate']
        self.poly_degree = hyperparam_dict['poly_degree']
        self.state_size = hyperparam_dict['state_size']
        self.feature_size = (self.poly_degree+1)**self.state_size
        self.activation_func = hyperparam_dict['activation_function']
        self.sample_func = hyperparam_dict['sample_function']
        self.sample_std = hyperparam_dict['sample_log_std']
        self.num_actions = hyperparam_dict['num_actions']
        
        self.affine1 =  nn.Linear(self.feature_size, self.num_actions)

        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        action_scores = self.affine1(x)
        return self.activation_func(action_scores)
    
    def select_action(self,feature):        
        # Convert the feature from a numpy array to a torch tensor
        feature = torch.from_numpy(feature).float().unsqueeze(0)
        
        # Get the predicted action from the policy network
        mu = self.forward(feature)
        
        mu, action, log_prob = self.sample_func(mu, act_dim=self.num_actions, log_std=self.sample_std)
        
        # Also save the log of the probability for the selected action
        self.saved_log_probs.append(log_prob)
        
        # Return the chosen action value
        return action

    def save(self, state_file='models/LinearFAPolicy.pt'):
        # Save the model state
        torch.save(self.state_dict(), state_file)

    @staticmethod
    def load(hyperparams, state_file='models/LinearFAPolicy.pt'):
        
        # Create a network object with the constructor parameters
        policy = LinearFAPolicy()
        policy.set_params(hyperparams)
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
        
if __name__ == "__main__":
    #feats = poly_feature(np.array([ 1.2303354,   1.5231776,   2.3200812,  -0.02065281,  0.23316315, -0.97967625,
  #0.28163096,  0.49523485]))
    for i in range(100):
        feats = poly_feature(np.array([ 1.2303354,   1.5231776,   2.3200812,  -0.02065281,  0.23316315, -0.97967625,
  0.28163096,  0.49523485]),3)
    print(feats.shape)
    #linear_fa_policy = LinearFAPolicy(feats.shape[0], num_actions=2, activation_function=torch.tanh, 
    #                                    sample_function = gaussian_policy)

    #print(linear_fa_policy.select_action(feats))

