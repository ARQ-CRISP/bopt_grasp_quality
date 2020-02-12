#!/usr/bin/env python
from __future__ import print_function, division

import rospy
import numpy as np
# from math import nan
from geometry_msgs.msg import Pose

from bopt_grasp_quality.srv import bopt, boptResponse
from bayesian_optimization import BOGrasp_Quality


class BO_Engine():

    def __init__(self, n, params, lp=None, up=None):
        rospy.init_node('ros_bo')
        rospy.on_shutdown(self.shutdown)

        rospy.wait_for_service('bayes_optimization')
        self.send_query = rospy.ServiceProxy('bayes_optimization', bopt)
        self.BO = BOGrasp_Quality(n, self.min_function, params, lp, up)
        mvalue, x_out, error = self.BO.optimize()
        res = self.send_query(Pose(), True, x_out, mvalue)
        while not rospy.is_shutdown():
            pass

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
    params['n_iterations'] = 50
    params['n_iter_relearn'] = 5
    params['n_init_samples'] = 5
    params['l_type'] = "L_MCMC"
    
    BO_Engine(n, params)