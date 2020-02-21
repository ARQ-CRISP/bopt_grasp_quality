#!/bin/python
from __future__ import division, print_function

from time import clock, sleep

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt import callbacks
from enum import Enum

def random_search(fun, n, bounds, params):
    space = [np.arange(bounds[i][0], bounds[i][0], params['diff'][i]) for i in range(n)]
    get_sample = lambda : np.array([np.random.choice(space[i]) for i in range(n)]) 
    Xi = get_sample()
    Yi = fun(Xi)

    min_point = (Xi, Yi)
    for itr in range(1, params['n_calls']):
        print('ITERATION {:d}'.format(itr))
        Xi = get_sample()

        Yi = fun(Xi)
        if min_point[1] > Yi:
             min_point = (Xi, Yi)
    
    return min_point

    

class Random_Explorer():
    def __init__(self, fun, n, params, lb, ub):
        self.n_dim = n
        lb = np.zeros((n,)) if lb is None else lb
        ub = np.ones((n,)) if ub is None else ub
        self.bounds = [(l,u) for l,u in zip(lb,ub)]
        self.min_fun = f
        self.model_params = dict() if params is None else params
        self.history_x = []
        self.history_y = []
        self.stopping_callbacks = []
        self.checkpoint_file = None
        self.deltaY = None
        self.deltaX = None
        self.set_defaults()
    
    def helper_fun(self, Xin):
        Y = self.min_fun(np.array(Xin))
        self.history_x += [Xin]
        self.history_y += [Y]
        
        return Y
    def set_defaults(self):
        self.model_params.setdefault('diff', [.1] * self.n_dim) 
        self.model_params.setdefault('init', [(lb + ub)/2 for lb, ub in self.bounds]) 
        self.model_params.setdefault('n_calls', 100) 

    def optimize(self):
        res = random_search(self.helper_fun, self.bounds, self.model_params)
        # res = gp_minimize(self.helper_fun, self.bounds, **self.model_params)
        self.min_value = (res.x[0], res.fun)
        # self.opt_result = res
        return self.min_value


if __name__ == "__main__":
        

    w = [1,5,4]
    p = np.poly1d(w)
    n = 1     

    f = lambda x: p(x) + 1 * np.sin(2 * np.pi * 3 * x) #+ np.random.normal(0.0, scale=.5)

    bo = Random_Explorer(n, f, lb=-3 * np.ones((1,)), ub=3 * np.ones((1,)))
    res = bo.optimize()
    x = np.arange(-5, 5, 0.01)
    y = f(x)

    plt.plot(x, y, 'r')
    plt.scatter(bo.history_x, bo.history_y, marker='x')
    plt.scatter(*bo.min_value, marker='o')
    # plt.draw()
    # plt.ioff()
    plt.show()

