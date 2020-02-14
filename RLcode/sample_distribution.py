import torch
import torch.distributions as torch_dist
import numpy as np

def gaussian_policy(mu, act_dim=2, log_std = -0.5):
    log_std = log_std * torch.ones(act_dim)  # Set a constant log std
    
    std = torch.exp(log_std)
 #   std = torch.tensor([0.1,0.6]) 
    dist = torch_dist.Normal(mu, std)
    action = dist.sample()  # Sample an action
    log_prob = dist.log_prob(action)  # Find the log probability of the action
    return mu, action, log_prob

def categorical_policy(probs,*_):
    m = torch_dist.Categorical(probs)
    action = m.sample()
    return action
    
def exploit_policy(action, *_):
    return action, action, 1.0
    
