import random
import numpy as np
import math
import torch
import torch.nn as nn

from sample_distribution import *


def poly_feature(states, degree=2):
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

def linear_fa_policy(**hyperparam_dict):
    """
    combines the feature vector linearly with weights to produce the action
    """
    gamma = hyperparam_dict['gamma']
    learning_rate = hyperparam_dict['learning_rate']
    poly_degree = hyperparam_dict['poly_degree']
    state_size = hyperparam_dict['state_size']
    feature_size = (poly_degree+1)**state_size

    linear_fa_policy = LinearFAPolicy(feature_size, num_actions=2, 
                                        activation_function=torch.tanh,
                                        sample_function=gaussian_policy)
    linear_fa_policy.train()
    
    return linear_fa_policy

class LinearFAPolicy(nn.Module):
    def __init__(self, feature_size, num_actions, activation_function, sample_function):
        super(LinearFAPolicy, self).__init__()
        # Degree of the polynomial feature
        self.activation_func = activation_function
        self.sample_func = sample_function
        self.num_actions = num_actions
        self.affine1 =  nn.Linear(feature_size, 2)

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
        
        mu, action, log_prob = self.sample_func(mu, act_dim=self.num_actions)
        
        # Also save the log of the probability for the selected action
        self.saved_log_probs.append(log_prob)
        
        # Return the chosen action value
        return action

    def save(self, state_file='models/LinearFAPolicy.pt'):
        # Save the model state
        torch.save(self.state_dict(), state_file)

    @staticmethod
    def load(state_file='models/LinearFAPolicy.pt'):
        
        # Create a network object with the constructor parameters
        policy = Policy()
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
        
if __name__ == "__main__":
    feats = poly_feature(np.array([ 1.2303354,   1.5231776,   2.3200812,  -0.02065281,  0.23316315, -0.97967625,
  0.28163096,  0.49523485]))
    print(feats.shape)
    linear_fa_policy = LinearFAPolicy(feats.shape[0], num_actions=2, activation_function=torch.tanh, 
                                        sample_function = gaussian_policy)

    print(linear_fa_policy.select_action(feats))
    print(linear_fa_policy.select_action(feats))

