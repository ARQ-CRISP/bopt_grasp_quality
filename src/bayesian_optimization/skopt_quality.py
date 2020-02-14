#!/bin/python
from __future__ import division, print_function

from time import clock, sleep

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize


class Skopt_BO():
    def __init__(self, n, f, params=None, lb=None, ub=None):
        self.n_dim = n
        lb = np.zeros((n,)) if lb is None else lb
        ub = np.ones((n,)) if ub is None else ub
        self.bounds = [(l,u) for l,u in zip(lb,ub)]
        self.min_fun = f
        self.model_params = dict() if params is None else params
        self.set_defaults()
        self.history_x = []
        self.history_y = []
        
        
    def helper_fun(self, Xin):
        Y = self.min_fun(np.array(Xin))
        self.history_x += [Xin]
        self.history_y += [Y]
        
        return Y

    def set_defaults(self):
        self.model_params.setdefault('base_estimator', None) 
        self.model_params.setdefault('n_calls', 50)
        self.model_params.setdefault('n_random_starts', 5) 
        self.model_params.setdefault('acq_func', 'gp_hedge' )
        self.model_params.setdefault('acq_optimizer', 'auto') 
        self.model_params.setdefault('x0', None) 
        self.model_params.setdefault('y0', None) 
        self.model_params.setdefault('random_state', None )
        self.model_params.setdefault('verbose', False )
        self.model_params.setdefault('callback', None )
        self.model_params.setdefault('n_points', 10000 )
        self.model_params.setdefault('n_restarts_optimizer', 1 )
        self.model_params.setdefault('xi', 0.01)
        self.model_params.setdefault('kappa', 1.96 )
        self.model_params.setdefault('noise', 'gaussian')
        self.model_params.setdefault('n_jobs', 1 )
        self.model_params.setdefault('model_queue_size', None)

    def optimize(self):
        res = gp_minimize(self.helper_fun, self.bounds, **self.model_params)
        self.opt_result = res
        self.min_value = (res.x[0], res.fun)
        return self.min_value



if __name__ == "__main__":
        

    w = [1,5,4]
    p = np.poly1d(w)
    n = 1     

    f = lambda x: p(x) + 1 * np.sin(2 * np.pi * 3 * x) #+ np.random.normal(0.0, scale=.5)

    bo = Skopt_BO(n, f, lb=-3 * np.ones((1,)), ub=3 * np.ones((1,)))
    res = bo.optimize()
    x = np.arange(-5, 5, 0.01)
    y = f(x)

    plt.plot(x, y, 'r')
    plt.scatter(bo.history_x, bo.history_y, marker='x')
    plt.scatter(*bo.min_value, marker='o')
    # plt.draw()
    # plt.ioff()
    plt.show()

