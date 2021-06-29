# -*- coding: utf-8 -*-
"""
@author: Shubham Subhnil
@description: A custom Mountain Car code for Bayesian Actor-Critic Reinforcement
Learning. BAC although being model-free, needs pre-conditioning of the domain parameters
over the environment physics boundaries.

How to: The Mountain car code initializes the current file and throws an instance
to the BAC and BAC_grad for computation. This is how we can keep the BAC standardized
accross all the environments. We will always need a custom BAC code for each
environmrnt in GYM.
"""
import numpy as np
import random

class mountain_car_v0:
    def __init__(self, env_params):
        #Initialize the environment variables and parameter functions.
        
    def dynamics(self, state, a_old, domain_params):
        x_old = state.x
        
        #Change domain_params.... to an initialized constant
        tmp3 = x_old(2) + (0.001 * a_old) - (0.0025 * cos(3 * x_old(1)))
        x(2) = max(domain_params.VEL_RANGE(1) , min(tmp3 , domain_params.VEL_RANGE(2)))
        
        tmp3 = x_old(1) + x(2)
        x(1) = max(domain_params.POS_RANGE(1) , min(+tmp3 , domain_params.POS_RANGE(2)))
        
        if (x(1) == domain_params.POS_RANGE(1)):
            x(2) = 0
        
        if (x(1) >= domain_params.GOAL):
            x(1) = domain_params.GOAL
            x(2) = 0
            
    def calc_score(self, theta, state, domain_params, _):
        y = state.y
        
        #feature values
        phi_x = zeros(domain_params.NUM_STATE_FEATURES,1)
        mu = zeros(domain_params.NUM_ACT,1)
        
        for tt in range(domain_params.NUM_STATE_FEATURES):
            tmp1 = y - domain_params.GRID_CENTERS(:,tt);
            phi_x(tt) = exp(-0.5 * np.transpose(tmp1) * domain_params.INV_SIG_GRID * tmp1)
            
        for tt in range(domain_params.NUM_ACT):
            if tt == 1:
                phi_xa = [phi_x; zeros(domain_params.NUM_STATE_FEATURES,1);]
            else:
                phi_xa = [zeros(domain_params.NUM_STATE_FEATURES,1); phi_x]
                
            mu(tt) = exp(np.transpose(phi_xa) * theta)
            
        mu = mu / sum(mu)
        
        tmp2 = rand########
        
        if tmp2 < mu(1):
            a = domain.params.ACT(1)
            scr = [phi_x * (1 - mu(1)); -phi_x * mu(2)]
        else:
            a = domain_params.ACT(2);
            scr = [-phi_x * mu(1); phi_x * (1 - mu(2))]
            
        return a, scr
    
    
    def calc_reward(self, state, _, _):
        reward = state.isgoal - 1
        return reward
    
    def is_goal(self, state, domain_params):
        
        if state.x(1) >= domain_params.GOAL:
            state.isgoal = 1
        else:
            state.isgoal = 0
            
        return state.goal
    
    def kernel_kx(self, state, statedic, domain_params):
        sigk_x = 1;
        ck_x = 1;
        x = np.transpose(state.x);
        xdic = np.transpose(vertcat(statedic.x));###################
        y = [domain_params.c_map_pos(1); domain_params.c_map_vel(1)] .* x;####################
        ydic = repmat([domain_params.c_map_pos(1); domain_params.c_map_vel(1)],1,size(xdic,2)) .* xdic ;#######################
        temp = pdist2(y',ydic').^2;#############
        kx = ck_x * exp(-temp / (2 * sigk_x*sigk_x));###########
        
        return kx
    
    def kernel_kxx(self, state, domain_params):
        kxx = 1
        return kxx
        
    def perf_eval(self, theta, domain_params, learning_params):
        step_avg = 0
        
        for l in range(domain_params.num_episode_eval):
            t = 0
            state = domain_params.random_state(domain_params)
            a, _ = domain_params.calc_score(theta, state, domain_params)
            
            while state.isgoal == 0 and t < learning_params.episode_len_max:
                for istep in range(domain_parameters.STEP):
                    if state.isgoal == 0:
                        state, _ = domain_params.dynamics(state, a, domain_params)
                        state = domain_params.is_goal(state, domain_params)
                a, _ = domain_params.calc_score(theta, state, domain_params, learning_params)
                t = t + 1
            
            step_avg = step_avg + t
        
        perf = step_avg / domain_params.num_episode_eval
        
        return perf
                
    def random_state(self, domain_params):
         x = [((domain_params.POS_RANGE(2) - domain_params.POS_RANGE(1)) * rand) + domain_params.POS_RANGE(1);
         ((domain_params.VEL_RANGE(2) - domain_params.VEL_RANGE(1)) * rand) + domain_params.VEL_RANGE(1)];
            
         y = [(domain_params.c_map_pos(1) * x(1)) + domain_params.c_map_pos(2);
                 (domain_params.c_map_vel(1) * x(2)) + domain_params.c_map_vel(2)];
        
         state.x = x;
         state.y = y;     
         state.isgoal = 0;



