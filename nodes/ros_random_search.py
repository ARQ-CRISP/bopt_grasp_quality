#!/usr/bin/env python
from __future__ import division, print_function

import numpy as np
import rospy
from rospkg.rospack import RosPack
from copy import deepcopy
from tf2_ros import TransformListener, Buffer
from bopt_grasp_quality.srv import bopt, boptResponse
from bayesian_optimization import Random_Explorer
from bayesian_optimization.opt_nodes import RS_Node
# from math import nan
from geometry_msgs.msg import PoseStamped, Pose, Transform

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

    # lb_y = rospy.get_param('~lb_x', -.2)
    # ub_y = rospy.get_param('~ub_x', .2)
    lb_x = [float(xx) for xx in rospy.get_param('~lb_x', [-.2, 0., -.2])]
    ub_x = [float(xx) for xx in rospy.get_param('~ub_x', [.2, 0., .2])]
    ee_link = rospy.get_param('~ee_link', 'hand_root')
    base_link = rospy.get_param('~base_link', 'world')
    service_name = rospy.get_param('~commander_service', 'bayes_optimization')
    n_iter = rospy.get_param('~search_iters', 20)
    resolution = rospy.get_param('~resolution', .001)

    tf_buffer = Buffer(rospy.Duration(50))
    tf_listener = TransformListener(tf_buffer)
    rospy.loginfo(rospy.get_name().split('/')[1] + ': Initialization....')
    rospy.loginfo(rospy.get_name().split('/')[1] + ': Getting current pose....')
    rospy.sleep(0.5)
    try:
        ARM_TF = tf_buffer.lookup_transform(base_link, ee_link, rospy.Time().now(), rospy.Duration(0.1))
        current_pose = TF2Pose(ARM_TF)
    except Exception as e:
        rospy.logerr('error in finding the arm...')
        rospy.logerr('Starting at (0, 0, 0), (0, 0, 0, 1)')

        current_pose = PoseStamped()
        current_pose.pose.orientation.w = 1.
        
    pose = [
        [current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z],
        [current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w]]
    rospy.loginfo(
        rospy.get_name().split('/')[1] + ': starting at: ({:.3f}, {:.3f}, {:.3f})-({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(*pose[0] + pose[1])
        )

    n = len(lb_x)
    init_pos = np.array([
        current_pose.pose.position.x, 
        current_pose.pose.position.y,
        current_pose.pose.position.z])
    assert(len(lb_x) == len(ub_x))
    params = {
        Random_Explorer.PARAMS.iters :n_iter,
        Random_Explorer.PARAMS.init_pos : init_pos,
        Random_Explorer.PARAMS.sampling : [resolution] * n}
    
    # lb = current_pose.pose.position.y + lb_x * np.ones((n,))
    # ub = current_pose.pose.position.y + ub_x * np.ones((n,))
    lb = init_pos[np.arange(len(lb_x))] + lb_x - 1e-10
    ub = init_pos[np.arange(len(ub_x))] + ub_x 
    
    RS_Node(n, params, lb=lb, ub=ub, init_pose=current_pose.pose, service_name=service_name)
