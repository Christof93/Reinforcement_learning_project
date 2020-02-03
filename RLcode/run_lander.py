import linear_policy 
import nn_policy
import random_agent
import gym
import sys

def run_lunar_lander(env, num_episodes, policy_function):
    """
    runs the lunar lander environment for a specific number of episodes 
    with a specified policy function.
    """
    rewards = []

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        while True:
            action = policy_function(observation)
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
    ## load environment
    env = gym.make('LunarLanderContinuous-v2')
    ## choose a policy
    policy = random_agent.get_action

    if len(sys.argv) > 1:
        run_lunar_lander(env, num_episodes=int(sys.argv[1]), policy_function = policy)
    else:
        run_lunar_lander(env, num_episodes=100, policy_function = policy)

