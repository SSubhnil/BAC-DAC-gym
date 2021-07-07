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
import random
import math
from scipy.spatial.distance import cdist

class mountain_car_v0:
    def __init__(self, observation_space, action_space, **kwargs):#Initialize Domain Parameters
        #Initialize the environment variables and parameter functions.
        self.POS_RANGE = np.array([observation_space.low[0], observation_space.high[0]], dtype=np.float32)
        self.VEL_RANGE = np.array([observation_space.low[-1], observation_space.high[-1]], dtype=np.float32)
        
        self.GOAL = self.POS_RANGE[-1]
        
        self.POS_MAP_RANGE = np.array([[0],[1]])
        self.VEL_MAP_RANGE = np.array([[0],[1]])
        self.GRID_SIZE = np.array([[4], [4]])
        
        #Features initialization
        self.c_map_pos = np.linalg.solve(np.array([[self.POS_RANGE[0], 1], [self.POS_RANGE[-1], 1]]),
                                         np.array([[self.POS_MAP_RANGE[0]],[self.POS_MAP_RANGE[-1]]]))
        self.c_map_vel = np.linalg.solve(np.array([[self.VEL_RANGE[0], 1],
                                                   [self.VEL_RANGE[-1], 1]]),
                                         np.array([[self.VEL_MAP_RANGE[0]],
                                                   [self.VEL_MAP_RANGE[-1]]]))
                                                                                        
        self.GRID_STEP = np.array([[(self.POS_MAP_RANGE[-1] - self.POS_MAP_RANGE[0])/self.GRID_SIZE[0]],
                                  [(self.VEL_MAP_RANGE[-1] - self.VEL_MAP_RANGE[0])/self.GRID_SIZE[-1]]])
        self.NUM_STATE_FEATURES = self.GRID_SIZE[0] * self.GRID_SIZE[-1]
        self.GRID_CENTERS = np.zeros((2,self.NUM_STATE_FEATURES), dtype = np.int32)
        
        for i in range(0, self.GRID_SIZE[0]):
            for j in range(0, self.GRID_SIZE[-1]):
                self.GRID_CENTERS[:, ((i-1)*self.GRID_SIZE[-1])+j] = np.array([
                    [self.POS_MAP_RANGE[0] + ((i- 0.5) * self.GRID_STEP[0])],
                    [self.VEL_MAP_RANGE[0] + ((j - 0.5) * self.GRID_STEP[1])]])
    
        self.sig_grid = 1.3 * self.GRID_STEP[0]
        self.sig_grid2 = self.sig_grid**2
        self.SIG_GRID = self.sig_grid2 * np.identity(2)
        self.INV_SIG_GRID = np.linalg.inv(self.SIG_GRID)
        self.phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        self.NUM_ACT = np.size(action_space)
        self.ACT = np.row_stack(np.arange(action_space.n))
        self.num_policy_param = self.NUM_STATE_FEATURES * self.NUM_ACT
        
        
    def dynamics(self, state, a_old, domain_params):
        #State is a 'list' with x, y and isgoal items. 
        x_old = state[0]
        x = np.zeros(2)
        #Change domain_params.... to an initialized constant
        tmp3 = x_old[1] + (0.001 * a_old) - (0.0025 * math.cos(3 * x_old[0]))
        x[1] = max(self.VEL_RANGE[1] , min(tmp3 , self.VEL_RANGE[2]))
        
        tmp3 = x_old[0] + x(2)
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
            
    def calc_score(self, theta, state, domain_params, _):
        y = state[1]#State is a 'list' with x, y and isgoal items. 
        
        #feature values
        phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        mu = np.zeros((self.NUM_ACT, 1))
        
        for tt in range(0, self.NUM_STATE_FEATURES):
            tmp1 = y - self.GRID_CENTERS[:,tt]
            #Turns out to be a scalar
            phi_x[tt, 0] = math.exp(-0.5 * np.transpose(tmp1) * self.INV_SIG_GRID * tmp1)
            
        for tt in range(0, self.NUM_ACT):
            if tt == 0:
                phi_xa = np.array([[phi_x],
                                   [np.zeros((self.NUM_STATE_FEATURES, 1))]])
            else:
                phi_xa = np.array([[(np.zeros(self.NUM_STATE_FEATURES, 1))],
                                   [phi_x]])
                
            mu[tt] = math.exp(np.transpose(phi_xa) * theta)
            
        mu = mu / sum(mu)
        
        tmp2 = np.random.uniform(low=0.0, high = 1.0)
        
        if tmp2 < mu[0]:
            a = self.ACT[0]
            scr = np.array([[phi_x * (1 - mu[0])],
                            [-phi_x * mu[1]]])
        else:
            a = self.ACT[1];
            scr = np.array([[-phi_x * mu[0]],
                            [phi_x * (1 - mu[1])]])
        
        #Have to look at how the a and scr are handled with 2D indices
        return a, scr
    
    
    def calc_reward(self, state, **kwargs):
        reward = state[0] - 1
        return reward
    
    def is_goal(self, state, domain_params):
        x = state[0]
        if x[0] >= self.GOAL:
            state[2] = 1
        else:
            state[2] = 0
            
        return state[2]
    
    def kernel_kx(self, state, statedic, domain_params):
        sigk_x = 1;
        ck_x = 1;
        x = np.transpose(state[0]);
        #Possible conflict at concatenation
        xdic = np.transpose(np.concatenate(i for i in statedic[:][0]))
        y = np.multiply(np.array([[self.c_map_pos[0]],
                                               [self.c_map_vel[0]]]), x)### We will see
        arbitrary = np.array([[self.c_map_pos[0]], [self.c_map_vel[0]]])
        ydic = np.multiply(np.matlib.repmat(arbitrary, 1, np.size(xdic,axis=1)), xdic)
        
        #Element-wise squaring of Euclidean pair-wise distance
        temp = cdist(np.transpose(y),np.transpose(ydic))**2
        kx = ck_x * math.exp(-temp / (2 * sigk_x*sigk_x))
        
        return kx
    
    def kernel_kxx(self, state, domain_params):
        kxx = 1
        return kxx
        
    def perf_eval(self, theta, domain_params, learning_params):
        step_avg = 0
        
        for l in range(self.num_episode_eval):
            t = 0
            state = self.random_state()
            a, _ = self.calc_score(theta, state, self)
            
            while state[2] == 0 and t < learning_params.episode_len_max:
                for istep in range(self.STEP):
                    if state[2] == 0:
                        state, _ = self.dynamics(state, a, self)
                        state = self.is_goal(state, self)
                a, _ = self.calc_score(theta, state, self, learning_params)
                t = t + 1
            
            step_avg = step_avg + t
        
        perf = step_avg / self.num_episode_eval
        
        return perf
                
    def random_state(self):
         x = np.array(
             [[((self.POS_RANGE[1] - self.POS_RANGE[0]) * np.random.uniform(low=0.0, high=1.0)) + self.POS_RANGE[0]],
              [((self.VEL_RANGE[1] - self.VEL_RANGE[0]) * np.random.uniform(low=0.0, high=1.0)) + self.VEL_RANGE[0]]]
             )
            
         y = np.array(
             [[(self.c_map_pos[0] * x[0]) + self.c_map_pos[1]],
              [(self.c_map_vel[0] * x[1]) + self.c_map_vel[1]]]
             )
         
         isgoal = 0
         #We will use "state" as a list object
         # state.x = x;
         # state.y = y;     
         # state.isgoal = 0;
         
         state = [x, y, isgoal]
         
         return state


