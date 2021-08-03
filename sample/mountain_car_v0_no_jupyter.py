"""
Code is the property of Shubham Subhnil. The allgorithm is referred from [see GitHub readme.md]
Please use the repository link and Author's name for presenting the code in academic and scientific works.

see env/bac_mountain_car.py, [root]/bac.m

DO NOT run the code. Only run each cell.
Find "MENU" header in cell 2.
Run cells 1 and 2 first.
Run cells 3 and 4 for sim_rendering and plotting respectively. 
"""

# Import Libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("..")

from env.BAC_mountain_car import mountain_car_continuous_v0 as mc0_env 
from BAC import BAC_main

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
get_existing_results = 1 ## 1 for importing existing results file
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

    if write_to_file == 0:
        perf.to_csv(r'results\MountainCar_BAC_Evaluation-(Reward_Fix-2).csv')
        pd_dataframe.to_csv(r'results\MountainCarContinuous-(Reward_Fix-2).csv')
    
else:
    """
    pd_dataframe: num_updates_max x 9
                  ["Episode batch", "Mean Absolute Error", "Mean Squared Error", "Learning Rate",
                   "Batchc Gym Reward", "Batch User Reward", "Avg. Episode Length", "BAC Gradient",
                   "Policy Evolution"]
    The last policy in the "Policy Evolution" is the learnt policy 'theta'
    """
    pd_dataframe = pd.read_csv('results\MountainCarContinuous.csv')
    # pd_dataframe = pd.read_csv('results\data_store.csv')
    perf = pd.read_csv('results\MountainCar_BAC_Evaluation.csv')
    
    theta = (pd_dataframe.loc[pd_dataframe.index[-1], 'Policy Evolution']).to_numpy() ## Final learned policy

# Uncomment to use the best policy yet vvv
#     theta = np.vstack([1.249851057,
# 2.353740868,
# 2.671234578,
# 1.855580538,
# 1.122794309,
# 2.42523414,
# 3.006517064,
# 2.231921839,
# -0.10364529,
# 0.409088984,
# 0.891831874,
# 0.848324243,
# -1.282159611,
# -1.823210222,
# -1.713346034,
# -1.037420735,
# -1.249851057,
# -2.353740868,
# -2.671234578,
# -1.855580538,
# -1.122794309,
# -2.42523414,
# -3.006517064,
# -2.231921839,
# 0.10364529,
# -0.409088984,
# -0.891831874,
# -0.848324243,
# 1.282159611,
# 1.823210222,
# 1.713346034,
# 1.037420735
# ])

    # theta = pd.read_csv('results\theta_final.csv') ## Final learned policy
    
#%%
# Visualize
# Apply the learned policy on the Gym render
random_state = env_main.reset()
env_main.render()
state_c_map = domain.c_map_eval(random_state)
a, _ = domain.calc_score(theta, state_c_map)
done = False
episode_length = 300


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

input("Sim done. Press enter to close...")
env_main.close()

#%% Data Plotting
plt.figure(0)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Mean Squared Error"]], 'b-',
         label = "MSE")
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Mean Absolute Error"]], 'r-',
         label = "MAE")
plt.xlabel("BAC Batch")
plt.ylabel("MSE and MAE")
plt.legend()

plt.figure(1)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Batch User Reward"]], 
          'ro')
plt.xlabel("BAC Batch")
plt.ylabel("Avg. Batch Reward")

plt.figure(2)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Avg. Episode Length (t)"]], 
         'g*')
plt.xlabel("BAC Batch")
plt.ylabel("Avg. Episode Length (t)")



