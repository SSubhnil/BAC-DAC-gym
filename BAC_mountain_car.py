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

class Domain:
    def __init__(self, domain_paramters):
        #Initialize the environment variables and parameter functions.
        
    def dynamics(self, state, a_old, domain_params):
        x_old = state.x;
        
        #Change domain_params.... to an initialized constant
        tmp3 = x_old(2) + (0.001 * a_old) - (0.0025 * cos(3 * x_old(1)));
        x(2) = max(domain_params.VEL_RANGE(1) , min(tmp3 , domain_params.VEL_RANGE(2)));
        
        tmp3 = x_old(1) + x(2);
        x(1) = max(domain_params.POS_RANGE(1) , min(+tmp3 , domain_params.POS_RANGE(2)));
        
        if (x(1) == domain_params.POS_RANGE(1)):
            x(2) = 0;
        
        if (x(1) >= domain_params.GOAL):
            x(1) = domain_params.GOAL;
            x(2) = 0;

