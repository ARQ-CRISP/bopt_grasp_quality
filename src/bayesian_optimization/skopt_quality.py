#!/bin/python
from __future__ import division, print_function

from time import clock, sleep

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt import callbacks
from enum import Enum


class Skopt_BO():
    class PARAMS(Enum):
        iters = 'n_calls'
        n_restarts = 'n_random_starts'
        acq_func = 'acq_func'
        acq_optimizer = 'acq_optimizer'
        surrogate = 'base_estimator'
        callbacks = 'callback'


    def __init__(self, n, f, params=None, lb=None, ub=None):
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

    def set_Xstopping_callback(self, delta):
        iscallback = lambda x: isinstance(x, callbacks.DeltaXStopper)
        current_callback = filter(iscallback, self.stopping_callbacks)
        if len(current_callback) == 0:
            self.stopping_callbacks.append(callbacks.DeltaXStopper(delta))
        else:
            current_callback[0].delta = delta
        self.deltaX = delta
    
    def set_Ystopping_callback(self, delta):
        iscallback = lambda x: isinstance(x, callbacks.DeltaYStopper)
        current_callback = filter(iscallback, self.stopping_callbacks)
        if len(current_callback) == 0:
            self.stopping_callbacks.append(callbacks.DeltaYStopper(delta))
        else:
            current_callback[0].delta = delta
        self.deltaY = delta
        
    def set_checkpointing(self, filepath):
        iscallback = lambda x: isinstance(x, callbacks.CheckpointSaver)
        current_callback = filter(iscallback, self.stopping_callbacks)
        if len(current_callback) == 0:
            self.stopping_callbacks.append(callbacks.CheckpointSaver(filepath, store_objective=False))
        else:
            current_callback[0].checkpoint_path = filepath
        self.checkpoint_file = filepath
        


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
        self.model_params.setdefault('callback', self.stopping_callbacks )
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

    def plot_result1D(self, n_samples=400):
        conf95 = 1.96
        res = self.opt_result
        fig = plt.figure()
        plt.ion()
        plt.title('Estimated Function')
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], n_samples).reshape(-1, 1)
        x_gp = res.space.transform(x.tolist())
        gp = res.models[-1]
        y_pred, sigma = gp.predict(x_gp, return_std=True)
        plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        plt.fill(
            np.concatenate([x, x[::-1]]), 
            np.concatenate([y_pred - conf95 * sigma, (y_pred + conf95 * sigma)[::-1]]), 
            alpha=.2, fc="g", ec="None")
            # Plot sampled points
        plt.plot(self.history_x, self.history_y,
                "r.", markersize=8, label="Observations")
        plt.xlabel('X position')
        plt.legend()
        plt.draw()
        plt.pause(0.1)

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

