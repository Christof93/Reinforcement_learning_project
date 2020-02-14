import linear_policy 
import nn_policy
import random_agent
from reinforce import train_lunar_lander
import gym
import sys
import random
import numpy as np
import sample_distribution
import torch
import datetime
 

def make_observation(env):
    # Information about observations
    # Observation is an array of 8 numbers
    # These numbers are usually in the range of -1 .. +1, but spikes can be higher
    print('Shape of observations: {}'.format(env.observation_space.shape))
    print('A few sample observations:')
    for i in range(5):
       print(env.observation_space.sample())
    # Information about actions
    # Action is two floats [main engine, left-right engines].
    # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    # Left-right: -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
    print(env.action_space.shape)
    print('A few sample actions:')
    for i in range(5):
       print(env.action_space.sample())

def save_results(hyperparams, ep_rewards, running_rewards):
    with open("results/"+datetime.now()[:-5]+"training_results.txt", "w") as outfile:
        hyperparams_string = ""
        for param,val in hyperparams.items():
            hyperparams_string+=param+" -> "+str(val)+"\n"
        outfile.write(hyperparams_string)
    outfile.write("\nEpisode rewards:\n-------------")
    for i,(ep_rep, ru_rep) in enumerate(zip(ep_rewards, running_rewards)):
        outfile.write("Episode "+i+" rewards: "+ep_rep, ru_rep)
            
def save_plots(ep_rewards, running_rewards):
    fig, ax = plt.subplots(3,1, figsize=(24,18))
    ax[0].plot(range(len(ep_rewards)), ep_rewards, color='#b39bc8', lw=2, label='episode rewards')
    ax[0].plot(range(len(running_rewards)), running_rewards, color='#a00000', lw=2, label='running rewards')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Cumulative reward')
    ax[0].set_title('Learned Agent', fontsize=16)
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(range(len(random_ep_rewards)), random_ep_rewards, color='#7ed9d9', lw=2, label='episode rewards')
    ax[1].plot(range(len(random_running_rewards)), random_running_rewards, color='#133264', lw=2, label='running rewards')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Cumulative reward')
    ax[1].set_title('Random Agent', fontsize=16)
    ax[1].legend()
    ax[1].grid()
    
    ax[2].plot(range(len(running_rewards)), running_rewards, color='#a00000', lw=2, label='learned agent running rewards')
    ax[2].plot(range(len(random_running_rewards)), random_running_rewards, color='#133264', lw=2, label='random agent running rewards')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Cumulative reward')
    ax[2].set_title('Comparison between Learned and Random Agent', fontsize=16)
    ax[2].legend()
    ax[2].grid()
    
    plt.show()
    
def run_random_lander(env, num_episodes, policy_function):
    """
    runs the lunar lander environment for a specific number of episodes 
    with a specified policy function.
    """
    rewards = []

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        while True:
            action = policy_function(env, observation)
            observation, reward, done, info = env.step(action)
            # You can comment the below line for faster execution
            env.render()
            episode_reward += reward
            if done:
                print('Episode: {} Reward: {}'.format(episode, episode_reward))
                rewards.append(episode_reward)
                break
        print('Average reward: %.2f' % (sum(rewards) / len(rewards)))

    
if __name__=="__main__":
    # Initialize the environment
    env = gym.make('LunarLanderContinuous-v2')  
    
    # Define hyperparameters
    RANDOM_SEED = 1
    # Set seeds for reproducability
    random.seed(RANDOM_SEED)  
    env.seed(RANDOM_SEED)  
    np.random.seed(RANDOM_SEED)

    ## choose a policy
    state_size = env.observation_space.sample().shape[0]
    num_actions = env.action_space.sample().shape[0]
    hyperparam_linFA = {
                        'name': 'linearFA', 'gamma':0.95, 'poly_degree':1, 
                        'learning_rate':5e-3, 'state_size':state_size,
                        'num_actions':num_actions, 'activation_function': torch.tanh,
                        'sample_function': sample_distribution.gaussian_policy,
                        'sample_log_std':-1.5
                        }

    random_policy = random_agent.get_action
    linear_fa_policy = linear_policy.linear_fa_policy(**hyperparam_linFA)
    

    if len(sys.argv) > 1:
        try:
            model_file = sys.argv[2]
            saved_policy = linear_fa_policy.load(hyperparam_linFA, model_file)
            ep_rewds, run_rewds = train_lunar_lander(env, render = False, log_interval = 10, 
                                max_episodes=int(sys.argv[1]), policy=saved_policy,
                                max_steps=8000, hyperparams=hyperparam_linFA)
    
        except IndexError:
            ep_rewds, run_rewds = train_lunar_lander(env, render = False, log_interval = 10, 
                                max_episodes=int(sys.argv[1]), policy=linear_fa_policy, 
                                hyperparams=hyperparam_linFA)
    
    else:
        run_random_lander(env, num_episodes=100, policy_function = random_policy)
        
    save_results(hyperparam_linFA, ep_rewds, run_rewds)
