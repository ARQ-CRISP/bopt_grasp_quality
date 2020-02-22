#!/bin/python
from __future__ import division, print_function

from time import clock, sleep

from pickle import dump
import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt import callbacks
from enum import Enum

def random_search(fun, bounds, params):
    n = len(bounds)
    space = [np.arange(bounds[i][0], bounds[i][1], params['diff'][i]) for i in range(n)]
    get_sample = lambda : np.array([np.random.choice(space[i]) for i in range(n)]) 

    if params['init'] is None: # if None get a random sample
        Xi = get_sample()
    else: 
        if len(params['init']) != n: # if the init value has not the same dimension as the bounds raise Exception
            raise Exception('Init value should have the same dimension as the bounds. init: {}, bounds: {}'.format(len(params['init']), n))
        Xi = params['init']
    
    Yi = fun(Xi)
    min_point = (Xi, Yi)
    min_iter = 0
    for itr in range(1, params['n_calls']):
        # print('ITERATION {:d}'.format(itr + 1))
        Xi = get_sample()

        Yi = fun(Xi)
        for ctype, callback in params['callbacks'].items():
            if ctype == 'checkpointer':
                callback()
        if min_point[1] > Yi:
             min_point = (Xi, Yi)
             min_iter = itr
    
    return min_point, min_iter

    

class Random_Explorer():

    class PARAMS(Enum):
        iters = 'n_calls'
        init_pos = 'init'
        sampling = 'diff'
        callbacks = 'callbacks'

    def __init__(self, n, fun, lb, ub, params=None):
        self.n_dim = int(n)
        lb = np.zeros((n,)) if lb is None else lb
        ub = np.ones((n,)) if ub is None else ub
        self.bounds = [(l,u) for l,u in zip(lb,ub)]
        self.min_fun = fun
        self.model_params = dict() if params is None else params
        self.history_x = []
        self.history_y = []
        self.__callbacks = {}
        self.checkpoint_file = None
        self.min_value = None
        # self.deltaY = None
        # self.deltaX = None
        self.set_defaults()
    
    def __save_experiment(self):
        with open(self.checkpoint_file, 'w') as ff: 
            experiment_resume = dict()
            experiment_resume['history'] = (self.history_x, self.history_y)
            experiment_resume['params'] = { key: value for key, value in self.model_params.items() if key is not 'callbacks'}
            experiment_resume['bounds'] = self.bounds
            experiment_resume['found_min'] = self.min_value
            dump(experiment_resume, ff)

    def set_checkpointing(self, filepath):

        self.checkpoint_file = filepath
        if 'checkpointer' not in self.__callbacks.keys():
            self.__callbacks['checkpointer'] = self.__save_experiment



    def helper_fun(self, Xin):
        Y = self.min_fun(np.array(Xin))
        self.history_x += [Xin.tolist()] if isinstance(Xin, np.ndarray) else [Xin]
        self.history_y += [Y]
        
        return Y

    def set_defaults(self):
        # self.model_params.setdefault(self.PARAMS.sampling.value, [.1] * self.n_dim) 
        # self.model_params.setdefault(self.PARAMS.init_pos.value, None) # means random init
        # self.model_params.setdefault(self.PARAMS.iters.value, 100) 
        self.model_params.setdefault(self.PARAMS.sampling.value, [.1] * self.n_dim) 
        self.model_params.setdefault(self.PARAMS.init_pos.value, None) # means random init
        self.model_params.setdefault(self.PARAMS.iters.value, 100) 
        self.model_params.setdefault(self.PARAMS.callbacks.value, self.__callbacks) 

    def optimize(self):
        res = random_search(self.helper_fun, self.bounds, self.model_params)
        # res = gp_minimize(self.helper_fun, self.bounds, **self.model_params)
        self.min_value = (res[0][0].tolist(), res[0][1]) if isinstance(res[0][0], np.ndarray) else (res[0][0], res[0][1])
        self.min_iter = res[1]
        if 'checkpointer' in self.__callbacks.keys():
            self.__save_experiment()
        
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
    plt.text(bo.min_value[0] + .1, bo.min_value[1] + 5, s='{}'.format(bo.min_iter))
    print('Minimum discovered at iteration {:d}'.format(bo.min_iter))
    # plt.draw()
    # plt.ioff()
    plt.show()

