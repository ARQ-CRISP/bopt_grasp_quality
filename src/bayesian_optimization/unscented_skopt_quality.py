#!/bin/python
from __future__ import division, print_function

from time import clock, sleep

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize, ugp_minimize, callbacks, dump
# from enum import Enum
from bayesian_optimization import Skopt_BO

class Skopt_UBO(Skopt_BO):
    class PARAMS(Skopt_BO.PARAMS):
        sigma_params = 'sigma_params'
        sigma_cov = 'sigma_cov'
        
    def __init__(self, n, f, ut_cov, sigma_params=None, params=None, lb=None, ub=None):
        super(Skopt_UBO, self).__init__(n, f, params=params, lb=lb, ub=ub)
        
        self.model_params[self.PARAMS.sigma_cov] = ut_cov
        if sigma_params is not None:
            self.model_params[self.PARAMS.sigma_params] = sigma_params
        
        
    def set_defaults(self):
        super(Skopt_UBO, self).set_defaults()
        self.model_params.setdefault('acq_func', 'UEI' )
        self.model_params.setdefault('acq_optimizer', 'UOI')
        self.model_params.setdefault('sigma_params', {'alpha': .3, 'beta': 2., 'kappa': .1})
        self.model_params.setdefault('sigma_cov', 1e-3) 
        
    def optimize(self):
        res = ugp_minimize(
            self.helper_fun, self.bounds,
            **self.model_params)
        
        self.opt_result = res
        self.min_value = (res.x, res.fun)
        if self.checkpoint_file is not None:
            dump(res, self.checkpoint_file, store_objective=False)
        return self.min_value
 

if __name__ == "__main__":
        

    w = [1,5,4]
    p = np.poly1d(w)
    n = 1     

    f = lambda x: (p(x) + 1 * np.sin(2 * np.pi * 3 * x)) #+ np.random.normal(0.0, scale=.5)
    params = {
        'xi': 1e-2, 
        'kappa': 1e-2}
    # bo = Skopt_BO(n, f, lb=-3 * np.ones((1,)), ub=3 * np.ones((1,)), params=params)
    # bo_res = bo.optimize()
    ubo = Skopt_UBO(n, f, lb=-3 * np.ones((1,)), ub=3 * np.ones((1,)), ut_cov=0.01, params=params)
    ubo_res = ubo.optimize()
    x = np.arange(-5, 5, 0.01)
    y = f(x)
    
    # plt.plot(x, y, 'r')
    # plt.scatter(bo.history_x, bo.history_y, marker='x')
    # plt.scatter(*bo.min_value, marker='o')
    # plt.figure()
    plt.plot(x, y, 'r')
    plt.scatter(ubo.history_x, ubo.history_y, marker='x', c='green')
    plt.scatter(*ubo.min_value, marker='o')
    # plt.draw()
    # plt.ioff()
    plt.show()

