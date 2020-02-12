#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Pose

from bopt_grasp_quality.srv import bopt, boptResponse


class Fun_Service_Node():

    def __init__(self, fun):
        rospy.init_node('function_gen')
        rate = rospy.Rate(30)
        self.fun = fun
        self.service = rospy.Service('bayes_optimization', bopt, self.manage_query)
        self.query_hist = []
        self.reply_hist = []
        self.min_value = None
        fig = plt.figure()
        plt.axis([-10, 10, -80, 80])
        # plt.axis([-1.5, 1.5, 0, 6])
        plt.ion()
        self.plot_fun()
        ax = fig.axes
        plt.pause(.001)
        while not rospy.is_shutdown():
            plt.cla()
            fig.sca(*ax)
            plt.axis([-10, 10, -80, 80])
            # fig.sca(fig.get_axes())
            self.plot_fun()
            self.plot_datapoints()
            plt.pause(.01)
            rate.sleep()


    def manage_query(self, query_msg):

        if not query_msg.foundmin:
            Y = self.min_function(query_msg.query)
        else:
            print("The minimum has been found", (query_msg.minX, query_msg.minY))
            Y = np.nan
        rospy.sleep(.5)
        return boptResponse(Y)
        
    def min_function(self, query):
        
        x = query.position.x
        self.query_hist.append(x)
        Y = self.fun(x) + np.random.normal(0.0, scale=.5) 
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
    p = np.poly1d(np.poly([-5, -3, 0, 3]))
    fun = lambda x: p(x) + 2 * np.cos(2 * np.pi * 5 * x)
    x = np.arange(-10, 10, 0.01)
    
    Fun_Service_Node(fun)
