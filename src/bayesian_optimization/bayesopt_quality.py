#!/bin/python
from __future__ import division, print_function

import numpy as np
from time import clock, sleep
import matplotlib.pyplot as plt

import bayesopt
from bayesoptmodule import BayesOptContinuous, BayesOptDiscrete

# def f(X):
#     total = 5.0
#     for value in X:
#         total = total + (value - 0.33) * (value - 0.33)
#     # a=input('press to continue...')
#     # print("wait 1 sec...")
#     # sleep(1)
#     return total





class BOGrasp_Quality(BayesOptContinuous):
    def __init__(self, n, f, params, lb=None, ub=None):
        super(BOGrasp_Quality, self).__init__(n)
        # print(self.parameters)
        self.parameters = params
        self.upper_bound = ub if ub is not None else np.ones((n,))
        self.lower_bound = lb if lb is not None else np.zeros((n,))
        self.history_x = np.zeros((0, n))
        self.history_y = np.zeros((0, 1))
        self.min_fun = f
        # plt.axis([-1.5, 1.5, 0, 6])
        # plt.ion()
        # plt.show()


    def testfunc(self, Xin):
        Y = self.min_fun(Xin)
        Y = np.array(Y)
        self.history_x = np.concatenate([self.history_x, Xin.reshape(1, self.n_dim)], axis=0)
        self.history_y = np.concatenate([self.history_y, Y.reshape(1, 1)], axis=0)
        # plt.scatter(bo.history_x, bo.history_y)
        # plt.draw()
        # plt.pause(0.001)
        # plt.cla()
        return Y
        
    def evaluateSample(self, Xin):
        return self.testfunc(Xin)









if __name__ == "__main__":
        
    params = {}

    params['n_iterations'] = 50
    params['n_iter_relearn'] = 5
    params['n_init_samples'] = 5
    params['l_type'] = "L_MCMC"

    w = [1,5,4]
    p = np.poly1d(w)
    n = 1                  # n dimensions
    # lb = np.zeros((n,))
    # ub = np.ones((n,))

    f = lambda x: p(x) + 1 * np.sin(2 * np.pi * 3 * x) + np.random.normal(0.0, scale=.5)
    bo = BOGrasp_Quality(
        n=n, 
        f=f,
        params=params, 
        lb=-5 * np.ones((n,)), 
        ub=5* np.ones((n,)))


    mvalue, x_out, error = bo.optimize()
    x = np.arange(-5, 5, 0.01)
    y = f(x)


    print("Result", mvalue, "at", x_out)
    # print(x.shape, y.shape)
    plt.scatter(bo.history_x, bo.history_y)
    text = np.arange(1, bo.history_x.shape[0]).astype(str).tolist()
    for xx, yy, tt in zip(bo.history_x.tolist(), bo.history_y.tolist(), text):
        plt.text(xx[0]+.1, yy[0]+.1, tt, fontdict={'fontsize': 10},)

    plt.plot(x, y, 'r')
    # print(text)
    plt.ioff()
    plt.show()
    # plt.pause(1.5)

    # print("Running time:", clock() - start, "seconds")