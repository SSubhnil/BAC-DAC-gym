"""
DO NOT run the code. Only run each cell.
Find "MENU" header.
Run cell 1 and 2 first.
Run cell 3 and 4 for sim_rendering and plotting respectively. 
"""

# Import Libraries
import gym
import cupy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("..")

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
        self.episode_len_max = 150 ## Length of each episode in sec (e.g. 1000, 3000).
        self.num_update_max = 500 ## Apply the policy update after 500 cycles of BAC (e.g. 25000)
        self.sample_interval = 25 ## Policy evaluation after every 25 policy updates (e.g. 1000)
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

#%%
"""
MENU
"""
get_existing_results = 0 ## 1 for importing existing results file
                         ## 0 for running new simulation
write_to_file = 1 ## Writes results to .csv files only if get_existing_results == 0

identifier = "(Reward_Fix)"

"""
Initialize MountainCar-v0 environment for BAC algogrithm
num_episode_eval = 100 ## Number of experiment repeats for entire horizontal axis
to obtain average results and confidence intervals
"""
# Make environment
env_main = gym.make("MountainCarContinuous-v0")
env_main.reset()

# Initialize the learning parameters
learning_params = learning_parameters()

# Initialize the BAC_domain which is the BAC_mountain_car.py
domain = mc0_env(env_main, num_episode_eval = 10)

# Start BAC with learning parameters, BAC_domain and GYM environment
BAC_module = BAC_main(env_main, domain, learning_params)

if get_existing_results == 0:
    
    perf, theta, pd_dataframe = BAC_module.BAC()
    # theta is the final learned policy
    
    # Write to a file
    # convert 'perf' to Dataframe
    if write_to_file == 1:
        perf.to_csv(r'results\MountainCar_BAC_Evaluation-(CuPy_Fix).csv')
        pd_dataframe.to_csv(r'results\MountainCarContinuous-(CuPy_Fix).csv')
    
else:
    pd_dataframe = pd.read_csv('results\MountainCarContinuous.csv')
    perf = pd.read_csv('results\MountainCar_BAC_Evaluation.csv')
    theta = (pd_dataframe.loc[pd_dataframe.index[-1], 'Policy Evolution']).to_numpy() ## Final learned policy
    print(theta)
    
#%%
# Visualize
# Apply the learned policy on the Gym render
random_state = env_main.reset()
state_c_map = domain.c_map_eval(random_state)
a, _ = domain.calc_score(theta, state_c_map)
done = False
episode_length = 100

# Render for num_update_max/sample_interval time i.e. 0-axis length of 'perf'
t = 0
while done == False or t < episode_length:# in secs
# We expect the agent to converge within minimum time steps with learned policy
# 'theta'
    x_now, _, done, _ = env_main.step(np.array([a]))
    state_c_map = domain.c_map_eval(x_now)
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
plt.plot(pd_dataframe[["Avg. Episode Length (t)"]], pd_dataframe[["Episode Batch"]],
         label='Average Episode Length (t)')





