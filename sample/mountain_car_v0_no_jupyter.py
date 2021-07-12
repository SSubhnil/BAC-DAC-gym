# Import Libraries
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

import sys
sys.path.append('BAC-DAC-Gym/')

from BAC import BAC_main
from env.BAC_mountain_car import mountain_car_v0 as mc0_env

# Make environment
env_main = gym.make("MountainCar-v0")
env_main.reset()


class learning_parameters:
    """
    A Class for easier handling of Learning Parameters
    Another alternative is to have a "list" object but it will take further
    processing to extract variables from that list. This Class must be standard for all
    BAC environemnts.
    """
    def __init__(self):
        self.episode_len_max = 200 ## Length of each episode in sec (e.g. 1000, 3000).
        self.num_update_max = 500 ## Apply the policy update after 500 cycles of BAC (e.g. 25000)
        self.sample_interval = 50 ## Policy evaluation after every 50 policy updates (e.g. 1000)
        self.num_trial = 1 ## Number of times the entire experiment is run
        
        self.gam = 0.99 ## Discount Factor
        self.num_episode = 5 ## Number of episodes for BAC_Gradient estimation
        # Gradient Estimate occurs in batches of episodes
        # Can use 5, 10, 15, 20... 40. This has minimal effect on the convergence.
        
        self.alp_init_BAC = 0.025 ## Initial learning rate
        self.alp_variance_adaptive = 0 ## Fixed variance. Change to 1 for adaptive variance
        self.alp_schedule = 0 ## Fixed learning rate. Change to 1 for adaptive 'alpha'
        self.alp_update_param = 500 ## Total number of policy updates
        
        self.SIGMA_INIT = 1
        

"""
Initialize MountainCar-v0 environment for BAC algogrithm
num_episode_eval = 100 ## Number of experiment repeats for entire horizontal axis
to obtain average results and confidence intervals
"""
# Initialize the learning parameters
learning_params = learning_parameters()

# Initialize the BAC_domain which is the BAC_mountain_car.py
BAC_domain = mc0_env(env_main, num_episode_eval = 100)

# Start BAC with learning parameters, BAC_domain and GYM environment
BAC = BAC_main(env_main, BAC_domain, learning_params)


