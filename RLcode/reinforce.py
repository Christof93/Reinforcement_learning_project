import torch
import torch.optim as optim

from itertools import count
from linear_policy import poly_feature

import numpy as np

def finish_episode(policy, optimizer, gamma):
    # Variable for the current return
    G = 0
    policy_loss = []
    returns = []

    # Define a small float which is used to avoid divison by zero
    eps = np.finfo(np.float32).eps.item()

    # Go through the list of observed rewards and calculate the returns
    for r in policy.rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)

    # Convert the list of returns into a torch tensor
    returns = torch.tensor(returns)

    # Here we normalize the returns by subtracting the mean and dividing
    # by the standard deviation. Normalization is a standard technique in
    # deep learning and it improves performance, as discussed in 
    # http://karpathy.github.io/2016/05/31/rl/
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Here, we deviate from the standard REINFORCE algorithm as discussed above
    for log_prob, G in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * G)

    # Reset the gradients of the parameters
    optimizer.zero_grad()

    # Compute the cumulative loss
    policy_loss = torch.cat(policy_loss).mean()

    # Backpropagate the loss through the network
    policy_loss.backward()

    # Perform a parameter update step
    optimizer.step()

    # Reset the saved rewards and log probabilities
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train_lunar_lander(env, policy, hyperparams, feature_function, render=False, log_interval = 100, max_episodes=3000, max_steps=10000):
    # To track the reward across consecutive episodes (smoothed)
    running_reward = 1.0
    optimizer = optim.Adam(policy.parameters(), lr=hyperparams["learning_rate"])

    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()

    # Start executing an episode (here the number of episodes is unlimited)
    for i_episode in count(1):

        # Reset the environment
        state, ep_reward = env.reset(), 0
        # For each step of the episode
        for t in range(1, max_steps):  
            feature_vector = feature_function(state, hyperparams["poly_degree"])

            # Select an action using the policy network
            action = policy.select_action(feature_vector)
            action = np.array(action[0])
            # Perform the action and note the next state and reward
            state, reward, done, _ = env.step(action)
            if render:
                env.render()

            # Store the current reward
            policy.rewards.append(reward)

            # Track the total reward in this episode
            ep_reward += reward

            if done:
                break

        # Update the running reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Store the rewards for plotting
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        # Perform the parameter update according to REINFORCE
        finish_episode(policy,optimizer,hyperparams['gamma'])

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        # Stopping criteria
        if running_reward > env.spec.reward_threshold:
            print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break
        if i_episode >= max_episodes:
            print('Max episodes exceeded, quitting.')
            break
    # Save the trained policy network
    return ep_rewards, running_rewards
