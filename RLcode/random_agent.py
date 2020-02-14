import gym
import random
import numpy as np
import sys


def get_action(env, observation):
    '''
    Function that takes random actions
    Arguments:
        observation: The state of the environment (array of 8 floats)
    Returns:
        randomly chosen action (array of 2 floats)
    '''
    
    return env.action_space.sample()

def train_random_lander(env, num_episodes, render=False):
    ep_rewards = list()
    running_rewards = list()
        
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        while True:
            action = get_action(env, observation)
            observation, reward, done, info = env.step(action)
            # You can comment the below line for faster execution
            if render:
                env.render()
            episode_reward += reward
            if done:
                #print('Episode: {} Reward: {}'.format(episode, episode_reward))
                ep_rewards.append(episode_reward)
                break
    return ep_rewards, []
    
if __name__=="__main__":
    print()
