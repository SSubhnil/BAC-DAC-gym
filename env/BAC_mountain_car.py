# -*- coding: utf-8 -*-
"""
@author: Shubham Subhnil
@description: A custom Mountain Car code for Bayesian Actor-Critic Reinforcement
Learning. BAC although being model-free, needs pre-conditioning of the domain parameters
over the environment physics boundaries.

How to: The Mountain car code is initialized as an object and is thrown as an instance
to the BAC and BAC_grad for computation. This is how we can keep the BAC standardized
accross all the environments. We will always need a custom BAC code for each
environmrnt in GYM.
"""
import numpy as np
from numpy import matlib as mb
import math
from scipy.spatial import distance
 
class mountain_car_continuous_v0:
    def __init__(self, gym_env, **kwargs):#Initialize Domain Parameters
        #Initialize the environment variables and parameter functions.
        self.gym_env = gym_env
        observation_space = gym_env.observation_space
        action_space = gym_env.action_space
        self.POS_RANGE = np.array([[observation_space.low[0]], [observation_space.high[0]]], dtype=np.float32)
        self.VEL_RANGE = np.array([[observation_space.low[-1]], [observation_space.high[-1]]], dtype=np.float32)
        
        for key, value in kwargs.items():
            if key == "num_episode_eval":
                self.num_episode_eval = value
    
        self.GOAL = self.POS_RANGE[-1][0]
        
        self.POS_MAP_RANGE = np.array([[0],[1]])
        self.VEL_MAP_RANGE = np.array([[0],[1]])
        self.GRID_SIZE = np.array([[4], [4]])
        
        # Features initialization
        # c_map_(pos/vel) are 2 x 1 vectors 
        self.c_map_pos = np.linalg.solve(np.array([[self.POS_RANGE[0][0], 1], [self.POS_RANGE[-1][0], 1]]),
                                         np.array([[self.POS_MAP_RANGE[0][0]],[self.POS_MAP_RANGE[-1][0]]]))
        self.c_map_vel = np.linalg.solve(np.array([[self.VEL_RANGE[0][0], 1],
                                                   [self.VEL_RANGE[-1][0], 1]]),
                                         np.array([[self.VEL_MAP_RANGE[0][0]],
                                                   [self.VEL_MAP_RANGE[-1][0]]]))
        
        self.GRID_STEP = np.array([[(self.POS_MAP_RANGE[-1][0] - self.POS_MAP_RANGE[0][0])/self.GRID_SIZE[0][0]],
                                  [(self.VEL_MAP_RANGE[-1][0] - self.VEL_MAP_RANGE[0][0])/self.GRID_SIZE[-1][0]]])
        self.NUM_STATE_FEATURES = self.GRID_SIZE[0][0] * self.GRID_SIZE[-1][0]
        self.GRID_CENTERS = np.zeros((2,self.NUM_STATE_FEATURES), dtype = np.float32)
        
        for i in range(0, self.GRID_SIZE[0][0]):
            for j in range(0, self.GRID_SIZE[-1][0]):
                self.GRID_CENTERS[0, (i*self.GRID_SIZE[-1][0])+j] = self.POS_MAP_RANGE[0][0] + (
                    (i - 0.5) * self.GRID_STEP[0][0])
                self.GRID_CENTERS[1, (i*self.GRID_SIZE[-1][0])+j] = self.VEL_MAP_RANGE[0][0] + (
                    (j - 0.5) * self.GRID_STEP[1][0])
    
        self.sig_grid = 1.3 * self.GRID_STEP[0]
        self.sig_grid2 = self.sig_grid**2
        self.SIG_GRID = self.sig_grid2 * np.identity(2)
        self.INV_SIG_GRID = np.linalg.inv(self.SIG_GRID)
        self.phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        self.NUM_ACT = 2 ## -1.0 and 1.0 since Continuous control problem
        self.ACT = np.array([action_space.low[0], action_space.high[0]])
        self.num_policy_param = self.NUM_STATE_FEATURES * self.NUM_ACT
        self.STEP = 1
        
        self.prng = np.random.RandomState()
        self.prng.seed(2)
        
    def calc_score(self, theta, state):
        """
        Description-----
        calc_score: Chooses the next action 'a' and computes the Fisher Information
        matrix score 'scr' for the mountain car domain.
        Parameters------
        theta: Current policy
        state: Current State = [x, y]
        Return-----
        a: Action to move to next state
        scr: Fisher Information matrix
        """  
        
        y = state[1]#State is a 'list' with x, y and isgoal items. 
        
        #feature values
        phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        mu = np.zeros(self.NUM_ACT)
        tmp1 = np.zeros((2, 1))
        
        for tt in range(0, self.NUM_STATE_FEATURES):
            tmp1 = y - self.GRID_CENTERS[:,tt].reshape((2, 1))
            #Turns out to be a scalar
            # We solve x'Ax by Matmul Ax then dot product of x and Ax
            arbi1 = np.dot(self.INV_SIG_GRID, tmp1)
            phi_x[tt, 0] = np.exp(-0.5 * np.dot(np.transpose(tmp1), arbi1)).item()
            
        for tt in range(0, self.NUM_ACT):
            if tt == 0:
                phi_xa = np.vstack((phi_x, np.zeros((self.NUM_STATE_FEATURES, 1)))) 
                
            else:
                phi_xa = np.vstack((np.zeros((self.NUM_STATE_FEATURES, 1)), phi_x))

            lol = np.dot(np.transpose(phi_xa), theta)
                
            mu[tt] = np.exp(lol.item())
            
        mu = mu / sum(mu)
        
        tmp2 = self.prng.rand()
        
        # Added some randomness is the action value. a * tmp2
        if tmp2 < mu[0]:
            a = self.ACT[0] #* tmp2
            scr = np.vstack((phi_x * (1 - mu[0]),
                            -phi_x * mu[1]))
    
        else:
            a = self.ACT[-1] #* tmp2
            scr = np.vstack((-phi_x * mu[0],
                            phi_x * (1 - mu[1])))#.reshape((1, len(phi_x)*2))
        
        # scr HAS TO BE a row-wise 2D array of size num_state_features x 1 
        return a, scr
    
    
    def kernel_kx(self, state, statedic, domain_params):
        sigk_x = 1
        ck_x = 1
        x = state[0]
        xdic = []
        # Possible conflict at concatenation
        for i in range(0, len(statedic)):
            xdic.append(statedic[i][0].reshape((2, 1))) ## The shape is v-important
        xdic = np.hstack(xdic)
        arbitrary = np.vstack([self.c_map_pos[0][0], self.c_map_vel[0][0]])
        y = np.multiply(arbitrary, x)### We will see
        ydic = np.multiply(mb.repmat(arbitrary, 1, np.shape(xdic)[1]),  xdic)
        # Element-wise squaring of Euclidean pair-wise distance
        #Need to install pdist python package 
        temp = distance.cdist(np.transpose(y), np.transpose(ydic)) ** 2
        kx = ck_x * np.exp((-1 * temp) / (2 * sigk_x*sigk_x))
        return np.squeeze(kx)
    
    def kernel_kxx(self, state, domain_params):
        kxx = 1
        return kxx
        
    def perf_eval(self, theta, domain_params, learning_params):
        """
        Evaluates the policy after every n(sample_interval) (e.g. 50) updates.
        See BAC.py for the function call protocol --> Find --> perf_eval
        """
        step_avg = 0
        
        for l in range(0, self.num_episode_eval):
            t = 0
            env_current_state = self.gym_env.reset()
            state = self.c_map_eval(env_current_state)
            # Since Gym.reset() only returns state = (position, velocity)
            # and we also need a C map for this state which is necessary for
            # BAC computation and is exclusive for each environment and observations
            
            done = False
            a, _ = self.calc_score(theta, state)
            reward2 = 0
            reward1 = 0
            while done == 0 or t < learning_params.episode_len_max:
                for istep in range(self.STEP):
                    if done == 0:
                        #state, _ = self.dynamics(state, a, self)
                        #state = self.is_goal(state, self)
                        state[0], reward, done, _ = self.gym_env.step(np.array([a])) ### Fix this array methods
                        state = self.c_map_eval(state[0])
                        state.append(done)
                        reward1 += reward ## Reward accumulated by Gym
                        reward2 -= 1 ## User defined reward
                a, _ = self.calc_score(theta, state)
                t = t + 1
            
            step_avg = step_avg + t
        
        perf = step_avg / self.num_episode_eval
        
        return perf, reward1, reward2
                
    def c_map_eval(self, x):
                    
        y = np.array(
            [[(self.c_map_pos[0][0] * x[0]) + self.c_map_pos[1][0]],
             [(self.c_map_vel[0][0] * x[1]) + self.c_map_vel[1][0]]]
            ).reshape((2, 1))
        # We will use "state" as a list object         
        state = [x.reshape((2, 1)), y]
        return state
    
    def dynamics(self, state, a_old, domain_params):
        """
        dynamics() is obsolete when using Gym.
        Use this for custom Mountain Car dynamics
        """
        #State is a 'list' with x, y and isgoal items. 
        x_old = state[0]
        x = np.zeros(2)
        #Change domain_params.... to an initialized constant
        tmp3 = x_old[1] + (0.001 * a_old) - (0.0025 * math.cos(3 * x_old[0]))
        x[1] = max(self.VEL_RANGE[1] , min(tmp3 , self.VEL_RANGE[2]))
        
        tmp3 = x_old[0] + x[1]
        x[0] = max(self.POS_RANGE[0] , min(+tmp3 , self.POS_RANGE[1]))
        
        if (x[0] == self.POS_RANGE[0]):
            x[1] = 0
        
        if (x[0] >= self.GOAL):
            x[0] = self.GOAL
            x[1] = 0
            
        y = np.array([[(self.c_map_pos[0] * x[0]) + self.c_map_pos[1]],
                      [self.c_map_vel[0] * x[1] + self.c_map_vel[1]]])
        isgoal = 0
        nstate = [x, y, isgoal]
        out = []
        
        return nstate, out
     
    def calc_reward(self, state):
        reward = state[2] - 1
        return reward
    
    def is_goal(self, state, domain_params):
        x = state[0]
        if x[0] >= self.GOAL:
            state[2] = 1
        else:
            state[2] = 0
            
        return state[2]



