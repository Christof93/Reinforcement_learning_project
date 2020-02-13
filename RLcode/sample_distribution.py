import torch
import torch.distributions as torch_dist

def gaussian_policy(mu, act_dim=2):
    log_std = - 0.5 * torch.ones(act_dim)  # Set a constant log std
    std = torch.exp(log_std)
    dist = torch_dist.Normal(mu, std)
    action = dist.sample()  # Sample an action
    print(action)
    log_prob = dist.log_prob(action)  # Find the log probability of the action
    return mu, action, log_prob

def categorical_policy(probs):
    m = torch_dist.Categorical(probs)
    action = m.sample()
    return action
    
def exploit_policy(action, act_dim):
    print(type(action))
    return action, action, 1.0
    
