# Import Libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import glob

# import sys
# sys.path.append('../BAC-DAC-Gym/')

from env.bac_mountain_car import mountain_car_continuous_v0 as mc0_env 
from bac import BAC_main


class learning_parameters(object):
    """
    A Class for easier handling of Learning Parameters
    Another alternative is to have a "list" object but it will take further
    processing to extract variables from that list. This Class must be standard for all
    BAC environments.
    """
    def __init__(self):
        self.episode_len_max = 200 ## Length of each episode in sec (e.g. 1000, 3000).
        self.num_update_max = 500 ## Apply the policy update after 500 cycles of BAC (e.g. 25000)
        self.sample_interval = 50 ## Policy evaluation after every 50 policy updates (e.g. 1000)
        self.num_trial = 1 ## Number of times the entire experiment is run
        
        self.gam = 0.99 ## Discount Factor (see BAC_grad for implementation)
        self.num_episode = 5 ## Number of episodes for BAC_Gradient estimation
        # Gradient Estimate occurs in batches of episodes
        # Can use 5, 10, 15, 20... 40. This has minimal effect on the convergence.
        
        self.alp_init_BAC = 0.025 ## Initial learning rate
        self.alp_variance_adaptive = 0 ## Fixed variance. Change to 1 for adaptive variance
        self.alp_schedule = 0 ## Fixed learning rate. Change to 1 for adaptive 'alpha'
        self.alp_update_param = 500 ## Total number of policy updates
        
        self.SHOW_EVERY_RENDER = 100
        self.SIGMA_INIT = 1


# Make environment
env_main = gym.make("MountainCarContinuous-v0")
env_main.reset()

"""
Initialize MountainCar-v0 environment for BAC algogrithm
num_episode_eval = 100 ## Number of experiment repeats for entire horizontal axis
to obtain average results and confidence intervals
"""
# Initialize the learning parameters
learning_params = learning_parameters()

# Initialize the BAC_domain which is the BAC_mountain_car.py
domain = mc0_env(env_main, num_episode_eval = 100)

# Start BAC with learning parameters, BAC_domain and GYM environment
BAC_module = BAC_main(env_main, domain, learning_params)
perf, theta, pd_dataframe = BAC_module.BAC()
# theta is the final learned policy

#%%
# Visualize
# Apply the learned policy on the Gym render
random_state = env_main.reset()
state_c_map = domain.c_map_eval(random_state)
a, _ = domain.calc_score(theta, state_c_map)
done = False
episode_length = 200

# Render for num_update_max/sample_interval time i.e. 0-axis length of 'perf'
t = 0
while done == False or t < episode_length:# in secs
# We expect the agent to converge within minimum time steps with learned policy
# 'theta'
    state_c_map[0], _, done, _ = env_main.step(np.array([a]))
    state_c_map = domain.c_map_eval(state_c_map[0])
    a, _ = domain.calc_score(theta, state_c_map)
    t += 1
    
    env_main.render()

env_main.close()

#%% Data Plotting

plt.plot(pd_dataframe[["Mean Squared Error"]], pd_dataframe[["Episode Batch"]],
         label='MSE with BAC batch episodes')
plt.plot(pd_dataframe[["Mean Absolute Error"]], pd_dataframe[["Episode Batch"]],
         label='MAE with BAC batch episodes')
plt.plot(pd_dataframe[["Batch Gym Reward"]], pd_dataframe[["Episode Batch"]],
         label='Reward evolution with BAC gradient estimation')
plt.plot(pd_dataframe[["Batch User Reward"]], pd_dataframe[["Episode Batch"]],
         label='User Defined Reward evolution')


#%% Write to a file
# convert 'perf' to Dataframe
perf.to_csv(r'results\MountainCar_BAC_Evaluation.csv')
pd_dataframe.to_csv(r'results\MountainCarContinuous.csv')


