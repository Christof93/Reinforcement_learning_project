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
from matplotlib import pyplot as plt
 

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

def save_results(timestamp, hyperparams, ep_rewards, running_rewards):
    with open("results/"+timestamp+"training_results.txt", "w") as outfile:
        hyperparams_string = ""
        for param,val in hyperparams.items():
            hyperparams_string+=param+" -> "+str(val)+"\n"
        outfile.write(hyperparams_string)
        outfile.write("\nEpisode rewards:\n-------------\n")
        for i,(ep_rep, ru_rep) in enumerate(zip(ep_rewards, running_rewards)):
            outfile.write("Episode "+str(i)+" rewards: "+str(ep_rep)+"   "+ str(ru_rep)+"\n")

def make_plot(ax, title, ep_rewards, random_ep_rewards=[]):
    ax.plot(range(len(ep_rewards)), ep_rewards, color='#a00000', lw=2, label='learned agent rewards')
    ax.plot(range(len(random_ep_rewards)), random_ep_rewards, color='#133264', lw=2, label='random agent rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative reward')
    ax.set_title(title, fontsize=24)
    ax.legend()
    ax.grid()


def task1(num_episodes):
    # Initialize the environment
    env = gym.make('LunarLanderContinuous-v2')  
    
    # Define hyperparameters
    state_size = env.observation_space.sample().shape[0]
    num_actions = env.action_space.sample().shape[0]
    hyperparam_linFA = {
                        'name': 'linearFA', 'gamma':0.95, 'poly_degree':1, 
                        'learning_rate':5e-3, 'state_size':state_size,
                        'num_actions':num_actions, 'activation_function': torch.tanh,
                        'sample_function': sample_distribution.gaussian_policy,
                        'sample_log_std':-0.5
                        }
    # Set different seeds for reproducability
    random_seeds = [1,2,3]
    fig, ax = plt.subplots(3,1, figsize=(24,18))
    
    for i, random_seed in enumerate(random_seeds):
        random.seed(random_seed)  
        env.seed(random_seed)  
        np.random.seed(random_seed)
        
        random_ep_rewds, random_run_rewds = random_agent.train_random_lander(env, num_episodes=num_episodes)

        ## choose a policy
        linear_fa_policy = linear_policy.linear_fa_policy(**hyperparam_linFA)

        ep_rewds, run_rewds = train_lunar_lander(env, policy=linear_fa_policy,
                                                 hyperparams=hyperparam_linFA,
                                                 feature_function=linear_policy.poly_feature,
                                                 render = False, log_interval = 10,
                                                 max_episodes=num_episodes)
        linear_fa_policy.save("models/LinearFAPolicy_seed"+str(random_seed)+".pt")
        make_plot(ax[i], "seed "+str(random_seed), ep_rewds, random_ep_rewds)
        time_now = str(datetime.datetime.now())[:-7].replace(" ","_")
        save_results("task1_seed"+str(random_seed)+"_"+time_now, hyperparam_linFA, ep_rewds, run_rewds)
        
    plt.savefig("results/task1_"+time_now+".png")
    
def task2(num_episodes):
    # Initialize the environment
    env = gym.make('LunarLanderContinuous-v2')

    # Define hyperparameters
    state_size = env.observation_space.sample().shape[0]
    num_actions = env.action_space.sample().shape[0]
    hyperparam_nn = {
                    'name': 'nn', 'gamma':0.95, 'poly_degree':1,
                    'learning_rate':5e-3, 'state_size':state_size,
                    'num_actions':num_actions, 'activation_function': torch.tanh,
                    'sample_function': sample_distribution.gaussian_policy,
                    'sample_log_std':-0.5
                   }
    # Set different seeds for reproducability
    random_seeds = [1,2,3]
    fig, ax = plt.subplots(3,1, figsize=(24,18))

    for i, random_seed in enumerate(random_seeds):
        random.seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

        random_ep_rewds, random_run_rewds = random_agent.train_random_lander(env, num_episodes=num_episodes)

        ## choose a policy
        nn_policy_instance = nn_policy.nn_policy_func(**hyperparam_nn)

        ep_rewds, run_rewds = train_lunar_lander(env, policy=nn_policy_instance,
                                                 hyperparams=hyperparam_nn,
                                                 feature_function=nn_policy.dummy_feature,
                                                 render = False, log_interval = 10,
                                                 max_episodes=num_episodes)
        nn_policy_instance.save("models/NN_seed"+str(random_seed)+".pt")
        make_plot(ax[i], "seed "+str(random_seed), ep_rewds, random_ep_rewds)
        time_now = str(datetime.datetime.now())[:-7].replace(" ","_")
        save_results("task2_seed"+str(random_seed)+"_"+time_now, hyperparam_nn, ep_rewds, run_rewds)

    plt.savefig("results/task2_"+time_now+".png")
    
def test_run(num_episodes):
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
                        'sample_log_std':-2.0
                        }

    fig, ax = plt.subplots(1,1, figsize=(24,10))
    hyperparam_nn = {
                        'name': 'nn', 'gamma':0.95, 'poly_degree':1,
                        'learning_rate':5e-3, 'state_size':state_size,
                        'num_actions':num_actions, 'activation_function': torch.tanh,
                        'sample_function': sample_distribution.gaussian_policy,
                        'sample_log_std':-0.5
                        }

    random_policy = random_agent.get_action
    linear_fa_policy = linear_policy.linear_fa_policy(**hyperparam_linFA)
    nn_policy_instance = nn_policy.nn_policy_func(**hyperparam_nn)
    
    if len(sys.argv) > 1:
        # try:
        #     model_file = sys.argv[2]
        #     saved_policy = linear_fa_policy.load(hyperparam_linFA, model_file)
        #
        #     ep_rewds, run_rewds = train_lunar_lander(env, policy=saved_policy,
        #                         hyperparams=hyperparam_linFA,
        #                         feature_function=linear_policy.poly_feature,
        #                         render = False, log_interval = 10,
        #                         max_episodes=int(sys.argv[1]),
        #                         max_steps=8000)
        #
        # except IndexError:
        #     ep_rewds, run_rewds = train_lunar_lander(env, policy=linear_fa_policy,
        #                         hyperparams=hyperparam_linFA,
        #                         feature_function=linear_policy.poly_feature,
        #                         render = False, log_interval = 10,
        #                         max_episodes=int(sys.argv[1]))


        try:
            model_file = sys.argv[3]
            saved_policy = nn_policy_instance.load(hyperparam_nn, model_file)

            ep_rewds, run_rewds = train_lunar_lander(env, policy=saved_policy,
                                                     hyperparams=hyperparam_nn,
                                                     feature_function=nn_policy.dummy_feature,
                                                     render = False, log_interval = 10,
                                                     max_episodes=int(sys.argv[1]),
                                                     max_steps=8000)

        except IndexError:
            ep_rewds, run_rewds = train_lunar_lander(env, policy=nn_policy_instance,
                                                     hyperparams=hyperparam_nn,
                                                     feature_function=nn_policy.dummy_feature,
                                                     render = False, log_interval = 10,
                                                     max_episodes=num_episodes)
    
    else:
        run_random_lander(env, num_episodes=100, policy_function = random_policy)
    nn_policy_instance.save()
    make_plot(ax, ep_rewds, run_rewds)
    time_now = str(datetime.datetime.now())[:-7].replace(" ","_")
    save_results(time_now, hyperparam_linFA, ep_rewds, run_rewds)
    plt.savefig("results/"+time_now+".png")
    
if __name__=="__main__":
    if sys.argv[1] == "plot_task1":
        task1(int(sys.argv[2]))
    elif sys.argv[1] == "plot_task2":
        task2(int(sys.argv[2]))
    elif sys.argv[1]=="test":
        test_run(int(sys.argv[2]))
    
