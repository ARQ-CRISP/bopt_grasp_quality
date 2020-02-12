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
        self.fun = fun
        self.service = rospy.Service('bayes_optimization', bopt, self.manage_query)
        rospy.spin()


    def manage_query(self, query_msg):

        if not query_msg.foundmin:
            Y = self.min_function(query_msg.query)
        else:
            print("The minimum has been found", (query_msg.minX, query_msg.minY))
            Y = np.nan

        return boptResponse(Y)
        
    def min_function(self, query):
        
        x = query.position.x
        Y = self.fun(x)
        return Y


    def shutdown(self):

        rospy.loginfo('Closing Fun Node....')
        rospy.sleep(.5)



if __name__ == "__main__":
    p = np.poly1d(np.poly([-3, 0, 5]))
    fun = lambda x: p(x) + np.random.normal(0.0, scale=.5) + 1.5 * np.cos(2 * np.pi * 2 * x)
    x = np.arange(-5, 5, 0.01)
    
    plt.figure()
    plt.ion()
    plt.show()
    plt.plot(x, fun(x))
    plt.draw()
    plt.pause(.01)
    Fun_Service_Node(fun)