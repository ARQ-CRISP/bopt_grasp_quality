#!/usr/bin/env python
from __future__ import division, print_function

import numpy as np
import rospy
from copy import deepcopy
from tf2_ros import TransformListener, Buffer
from bayesian_optimization import BayesOpt_BO, Skopt_BO
from bopt_grasp_quality.srv import bopt, boptResponse
# from math import nan
from geometry_msgs.msg import PoseStamped, Pose, Transform


class BO_Node():

    def __init__(self, n, params, lb=None, ub=None, init_pose = Pose()):
        
        rospy.on_shutdown(self.shutdown)
        rate = rospy.Rate(30)
        self.init_pose = init_pose
        self.init_messages(lb, ub, params)
        rospy.wait_for_service('bayes_optimization')
        self.send_query = rospy.ServiceProxy('bayes_optimization', bopt)
        self.optimizer = Skopt_BO(n, self.min_function, params, lb=lb, ub=ub)
        try:
            x_out, mvalue = self.optimizer.optimize()
            res = self.send_query(Pose(), True, x_out, mvalue) # here the optimization has finished
            rospy.loginfo('Minimum has been reached: ({:.3f}, {:.3f})'.format(x_out, mvalue))
        except rospy.ServiceException as e:
            rospy.logerr('The Service for bayesian optimization has been Terminated prematurely.')
            rospy.logerr('Terminating the node...')
            exit()
        while not rospy.is_shutdown():
            rate.sleep()

    def init_messages(self, lb, ub, params):
        rospy.loginfo(rospy.get_name().split('/')[1] + ': Optimization bounds')
        rospy.loginfo(rospy.get_name().split('/')[1] + ': lower bounds {}'.format(lb.round(3)))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': upper bounds {}'.format(ub.round(3)))
            
            

    def min_function(self, Xin):
        p = deepcopy(self.init_pose)
        p.position.x = float(Xin)
        rospy.loginfo('Estimating metric at ({:.3f}, {:.3f}, {:.3f}) ...'.format(p.position.x, p.position.y, p.position.z))
        res = self.send_query(p, False, np.nan, np.nan)
        rospy.loginfo('Estimatation of the metric obtained: {:.3f}'.format(res.Y))
        # res= boptResponse()
        return res.Y


    def shutdown(self):

        rospy.loginfo('Closing ROS-BO Node....')
        rospy.sleep(.5)

def TF2Pose(TF_msg):

    new_pose = PoseStamped()
    new_pose.header = TF_msg.header
    new_pose.pose.position.x = TF_msg.transform.translation.x
    new_pose.pose.position.y = TF_msg.transform.translation.y
    new_pose.pose.position.z = TF_msg.transform.translation.z

    new_pose.pose.orientation.x = TF_msg.transform.rotation.x
    new_pose.pose.orientation.y = TF_msg.transform.rotation.y
    new_pose.pose.orientation.z = TF_msg.transform.rotation.z
    new_pose.pose.orientation.w = TF_msg.transform.rotation.w

    return new_pose


if __name__ == "__main__":
    rospy.init_node('ros_bo')

    tf_buffer = Buffer(rospy.Duration(50))
    tf_listener = TransformListener(tf_buffer)
    rospy.loginfo(rospy.get_name().split('/')[1] + ': Initialization....')
    rospy.loginfo(rospy.get_name().split('/')[1] + ': Getting current pose....')
    rospy.sleep(0.5)
    try:
        ARM_TF = tf_buffer.lookup_transform('world', 'hand_root', rospy.Time().now(), rospy.Duration(0.1))
        current_pose = TF2Pose(ARM_TF)
    except Exception as e:
        rospy.logerr('error in finding the arm...')
        rospy.logerr('Starting at (0,0,0), (0,0,0,1)')
        current_pose = PoseStamped()
        current_pose.pose.orientation.w = 1.
        
    pose = [
        [current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z],
        [current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w]]
    rospy.loginfo(rospy.get_name().split('/')[1] + ': starting at: ({:.3f},{:.3f},{:.3f})-({:.3f},{:.3f},{:.3f},{:.3f})'.format(*pose[0] + pose[1]))

    params = {}
    n = 1
    lb = current_pose.pose.position.x - .2 * np.ones((n,))
    ub = current_pose.pose.position.x + .2 * np.ones((n,))
    
    BO_Node(n, params, lb= lb, ub=ub, init_pose=current_pose.pose)
