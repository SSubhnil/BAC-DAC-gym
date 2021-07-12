# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 18:58:44 2021

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix



class BAC_main:
    
    def __init__(self, gym_env, domain, learning_params):
        self.gym_env = gym_env
        self.domain = domain
        self.learnin_params = learning_params
        self.BAC(domain, learning_params, gym_env)
        
    #Bayesian Actor-Critic function
    def BAC(self, d, learning_params, **kwargs):
        learning_params.num_output = (learning_params.num_update_max / learning_params.sample_interval) + 1
        perf = np.zeros((learning_params.num_trial, learning_params.num_output))
        
        #Have to add ~isfield(d, 'STEP') equivalent
        if d.STEP:
            d.STEP = 1
        
        for i in range(0, learning_params.num_trial):
            
           # exptime = now#Add current time module
            #toc
            
            #Add file handling protocol
            
            theta = np.zeros((d.num_policy_params, 1))
            
            #Fix the following expression
            alpha_schedule = learning_params.alp_init_BAC * (learning_params.alp_update_param / 
                                                                 (learning_params.alp_update_param + 
                                                                  (np.arange(1,(learning_params.num_update_max + 1)) - 1)))
            
            for j in range(0, learning_params.num_update_max + 1):
                
                # Policy evaluation after every n(sample_interval) policy updates
                if (j % (learning_params.sample_interval)) == 0:
                    evalpoint = math.floor((j+1) / learning_params.sample_interval) + 1
                    perf[i, evalpoint] = d.perf_eval(theta, d, learning_params)
                    
                    ### insert file handling protocol
                    
                G = csr_matrix((d.num_policy_param, d.num_policy_param), dtype = np.float32)
                
                for l in range(1, learning_params.num_episode+1):
                    t = 0
                    episode_states = []
                    episode_scores = []
                    
                    state = d.random_state(d)## Probably remove this
                    a, scr = d.calc_score(theta, state, d, learning_params)
                    scr = csr_matrix(scr)
                    
                    #state is a "list" object with x, y and isgoal elements
                    while state[2] == True and t < learning_params.episode_len_max:
                        
                        for istep in range(1, d.STEP):
                            if state[2] == 0:
                                state, _ = d.dynamics(state, a, d)
                                state = d.is_goal(state, d)
                        
                        G = G + (scr * scr.transpose())
                        for s in np.nditer(state):
                            episode_states[len(episode_states):] = s
                        for sco in np.nditer(scr):
                            episode_scores[len(episode_scores):] = sco
                            #OR
                            #episode_scores[len(episode_scores):] = [sco for sco in np.nditer(scr)]
                        
                        a, scr = d.calc_score(theta, state, d, learning_params)
                        
                        scr = csr_matrix(scr)
                        
                        t = t + 1
                    
                    episodes = (episode_states, episode_scores, t)
                
                #Fix the identity matrix
                G = G + 1e-6 * np.identity(G.shape[0])
                grad_BAC = self.BAC_grad(episodes, G, d, learning_params)
                
                if learning_params.alp_schedule:
                    alp = alpha_schedule[j]
                else:
                    alp = learning_params.alp_init_BAC
                
                theta = theta + (alp * grad_BAC)
    
        return perf, theta
    
    #Bayesian Actor=Critic Gradient Function
    def BAC_grad(self, episodes, G, domain_params, learning_params):
        gam = learning_params.gam
        gam2 = gam**2
        
        nu = 0.1
        sig = 3
        sig2 = sig**2
        ck_xa = 1
        sigk_x = 1.3 * 0.25
        
        #Initialization
        T = 0
        m = 0
        statedic = []
        scr_dic = []
        invG_scr_dic = []
        alpha = []
        C = []
        Kinv = []
        k = []
        c = []
        z = []
        
        for l in range(1, learning_params.num_episode):
            ISGOAL = 0
            t = 1
            T = T + 1
            c = np.zeros((m, 1))
            d = 0
            s = math.inf
            #################################################
            state = episodes[0][:, t]
            scr = episodes[1][:, t]
            scrT = csr_matrix.transpose(scr)
            
            temp1 = domain_params.state_kernel_kxx(state, domain_params)
            invG_scr = np.linalg.solve(G, scr)
            
            temp2 = ck_xa * (scrT * invG_scr)
            kk = temp1 + temp2
            
            if m > 0:
                k = ck_xa * (scrT * invG_scr_dic) #State-action kernel -- Fisher Information Kernel
                k = k + domain_params.state_kernel_kx(state, statedic, domain_params)
                kT = np.linalg.transpose(k)
                
                a = Kinv * kT
                delta = kk - (k * a)
            else:
                k = []
                a = []
                delta = kk
                
            if m == 0 or delta > nu:
                a_hat = a
                a_hatT = np.linalg.transpose(a_hat)
                #Treat all of them as array 'list'
                h = [[a], [-gam]]
                a = [[z], [1]]
                alpha = [[alpha], [0]]
                C = [[C, z], [np.linalg.transpose(z), 0]]
                Kinv = np.linalg.solve([[(delta * Kinv) + (a_hat * a_hatT), -a_hat],
                        [-a_hatT , 1]], delta)
                z = [[z], [0]]
                c = [[c], [0]]
                for s in np.nditer(state):
                    statedic[len(statedic):] = s
                
                ################Change scr to 1D array in BAC_mountain_car
                for dic in np.nditer(scr):
                    scr_dic[len(scr_dic):] = dic
                for scd in np.nditer(invG_scr):
                    invG_scr_dic[len(invG_scr_dic):] = scd
                m = m + 1
                k = [[kT], [kk]]
        
            #Time-loop
            while (t < episodes[2]):
                state_old = state
                k_old = k
                kk_old = kk
                a_old = a
                c_old = c
                s_old = s
                d_old = d
                
                r = domain_params.calculate_reward(state_old, -1, domain_params)
                
                coef = (gam * sig2) / s_old
                
                #Goal update
                if ISGOAL == 1:
                    dk = k_old
                    dkT = np.transpose(dk)
                    dkk = kk_old
                    h = a_old
                    c = (coef * c_old) + h- (C * dk)
                    s = sig2 - (gam * sig2 * coef) + (dkT * (c + (coef * c_old)))
                    d = (coef * d_old) + r - (dkT * alpha)
                
                #Non-goal update    
                else:
                    state = episodes[0][:, t + 1]
                    scr = episodes[1][:, t + 1]
                    scrT = csr_matrix.transpose(scr)
                    
                    if state[2]:
                        ISGOAL = 1
                        t = t-1
                        T = T-1
                    
                    temp1 = domain_params.state_kernel_kxx(state, domain_params)
                    invG_scr = np.linalg.solve(G, scr)
                    temp2 = ck_xa * (scrT * invG_scr)
                    kk = temp1 + temp2
                    
                    k = ck_xa * (scrT * invG_scr)
                    k = k + domain_params.state_kernel_kx(state, statedic, domain_params)
                    
                    a = Kinv * k
                    delta = kk - (np.transpose(k) * a)
                    
                    dk = k_old - (gam * k)
                    d = (coef * d_old) + r - (np.transpose(dk) * alpha)
                    
                    if delta > nu:
                        h = [a_old, -gam]
                        dkk = (np.transpose(a_old) * (k_old - (2 * gam * k))) + (gam2 * kk)
                        c = (coef * [c_old, 0]) + h - [C * dk, 0]
                        s = ((1 + gam2) * sig2) + dkk - (np.transpose(dk) * C * dk) + (2 * coef * np.transpose(c_old) * dk) - (gam * sig2 * coef)
                        alpha = [alpha, 0]
                        C = [[C, z], [np.transpose(z), 0]]
                        for s in np.nditer(state):
                            statedic[len(statedic):] = s
                        
                        ################Change scr to 1D array in BAC_mountain_car
                        for dic in np.nditer(scr):
                            scr_dic[len(scr_dic):] = dic
                        for scd in np.nditer(invG_scr):
                            invG_scr_dic[len(invG_scr_dic):] = scd
                        m = m + 1
                        Kinv = np.multiply([[(delta * Kinv) + (a * np.transpose(a)), -a],
                                [np.transpose(-a)                      , 1]], 1/delta)
                        a = [[z], [1]]
                        z = [[z], [0]]
                        k = [[k], [kk]]
                        
                    else:#delta <= nu
                        h = a_old - (gam * a)
                        dkk = np.transpose(h) * dk
                        c = (coef * c_old) + h - (C * dk)
                        s = ((1-gam2) * sig2) + (np.transpose(dk) * (c + (coef * c_old))) - (
                            gam * sig2 * coef)
                
                #Alpha update
                alpha = alpha + (c * (d / s))
                #C update
                C = C + (c * (np.transpose(c) / s))
                
                #Update time counters
                t = t + 1
                T = T + 1
        
        #For all the fuss we went through, FINALLY!
        grad = ck_xa * (scr_dic * alpha)
        
        return grad
                            
                    
                        
                    
        