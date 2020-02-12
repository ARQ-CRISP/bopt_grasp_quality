#!/usr/bin/env python
from __future__ import print_function, division

import rospy
import numpy as np
# from math import nan
from geometry_msgs.msg import Pose

from bopt_grasp_quality.srv import bopt, boptResponse
from bayesian_optimization import BOGrasp_Quality
from bayesian_optimization import BO_Grasp_Quality


class BO_Engine():

    def __init__(self, n, params, lb=None, ub=None):
        rospy.init_node('ros_bo')
        rospy.on_shutdown(self.shutdown)
        rate = rospy.Rate(30)
        rospy.wait_for_service('bayes_optimization')
        self.send_query = rospy.ServiceProxy('bayes_optimization', bopt)
        self.BO = BO_Grasp_Quality(n, self.min_function, params, lb=lb, ub=ub)
        x_out, mvalue = self.BO.optimize()
        res = self.send_query(Pose(), True, x_out, mvalue)
        rospy.loginfo('Minimum has been reached: ({}, {})'.format(x_out, mvalue))
        while not rospy.is_shutdown():
            rate.sleep()

    def min_function(self, Xin):
        p = Pose()
        p.position.x = Xin
        res = self.send_query(p, False, np.nan, np.nan)
        # res= boptResponse()
        return res.Y


    def shutdown(self):

        rospy.loginfo('Closing ROS-BO Node....')
        rospy.sleep(.5)



if __name__ == "__main__":

    params = {}
    n = 1
    # params['n_iterations'] = 50
    # params['n_iter_relearn'] = 5
    # params['n_init_samples'] = 5
    # params['l_type'] = "L_MCMC"
    
    BO_Engine(n, params, lb= -5*np.ones((n,)), ub=5*np.ones((n,)))