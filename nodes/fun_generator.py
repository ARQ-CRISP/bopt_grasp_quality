#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Pose

from bopt_grasp_quality.srv import bopt, boptResponse


class Fun_Service_Node():

    def __init__(self, fun, noise_scale=0.01):
        rospy.init_node('function_gen')
        rate = rospy.Rate(30)
        self.noise_scale = noise_scale
        self.fun = fun
        self.service = rospy.Service('bayes_optimization', bopt, self.manage_query)
        self.query_hist = []
        self.reply_hist = []
        self.processed_samples = 0
        self.min_value = None
        fig = plt.figure()
        axis = [-10, 10, -80, 80]
        plt.axis(axis)
        # plt.axis([-5, 5, -5, 5])
        # plt.axis([-1.5, 1.5, 0, 6])
        plt.ion()
        self.plot_fun()
        ax = fig.axes
        plt.pause(.001)
        while not rospy.is_shutdown():
            # fig.sca(fig.get_axes())
            if self.processed_samples < len(self.reply_hist):
                # fig.sca(*ax)
                # plt.axis(axis)
                plt.clf()
                self.plot_fun()
                self.plot_datapoints()
                self.processed_samples += 1
            plt.pause(.01)
            rate.sleep()


    def manage_query(self, query_msg):

        if not query_msg.foundmin:
            Y = self.min_function(query_msg.query)
        else:
            print("The minimum has been found x={:.3f}, Gm={:.3f}".format(query_msg.minX, query_msg.minY))
            Y = np.nan
            self.min_value = [query_msg.minX, query_msg.minY]
        rospy.sleep(.2)
        return boptResponse(Y)
        
    def min_function(self, query):
        
        x = query.position.x
        Y = self.fun(x) + np.random.normal(0.0, scale=self.noise_scale) 
        self.query_hist.append(x)
        self.reply_hist.append(Y)
        return Y


    def shutdown(self):

        rospy.loginfo('Closing Fun Node....')
        rospy.sleep(.5)

    def plot_datapoints(self):

        if len(self.query_hist) > 0:
            plt.scatter(self.query_hist, self.reply_hist, marker='x')
            plt.draw()
            text = np.arange(1, len(self.query_hist)).astype(str).tolist()
            for xx, yy, tt in zip(self.query_hist, self.reply_hist, text):
                plt.text(xx + .1, yy + .1, tt, fontdict={'fontsize': 10},)

        if self.min_value is not None:
            plt.scatter(*self.min_value, marker='o')


    def plot_fun(self):
        # plt.clf()
        plt.plot(x, self.fun(x), 'r')
        # plt.scatter(, )
        plt.draw()
        # plt.pause(.01)
        


if __name__ == "__main__":
    from scipy.stats import norm
    p = np.poly1d(np.poly([-5, -3, 0, 3]))
    s = .2
    m = 0
    # gaussian = lambda x: ((1/(s*np.sqrt(2*np.pi))))*np.exp(-((x - m)/s)**2)
    gaussian = lambda x : norm.pdf(x, m, s)

    # fun = lambda x: - gaussian(x)
    fun = lambda x: p(x) + 2 * np.cos(2 * np.pi * 5 * x)
    x = np.arange(-10, 10, 0.01)
    
    Fun_Service_Node(fun, noise_scale=0.5)
