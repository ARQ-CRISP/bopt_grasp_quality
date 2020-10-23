#!/usr/bin/env python
from __future__ import division, print_function

import numpy as np
import rospy
from rospkg.rospack import RosPack
from copy import deepcopy
from tf2_ros import TransformListener, Buffer
# from bayesian_optimization import BayesOpt_BO
from bayesian_optimization import Skopt_BO
from bopt_grasp_quality.srv import bopt, boptResponse
# from math import nan
from geometry_msgs.msg import PoseStamped, Pose, Transform


class BO_Node():

    __package_name = 'bopt_grasp_quality'
    def __init__(self, n, params, lb=None, ub=None, init_pose = Pose(), service_name='bayes_optimization', checkpoint='BayesOpt.pkl'):
        
        rospy.on_shutdown(self.shutdown)
        pack_path = RosPack().get_path(self.__package_name)
        rate = rospy.Rate(30)
        if ub is not None or lb is not None:
            self.regr_vars = np.where((np.array(ub) - np.array(lb)) >= 1e-6)[0]
        else:
            self.regr_vars = np.arange(n)
        self.init_pose = init_pose
        self.optimizer = Skopt_BO(len(self.regr_vars), self.min_function, params, lb=lb[self.regr_vars], ub=ub[self.regr_vars])
        self.optimizer.set_Xstopping_callback(1e-3)
        if checkpoint is not None:
            self.optimizer.set_checkpointing(pack_path + '/etc/' + checkpoint)
        self.init_messages(lb, ub, params)
        rospy.wait_for_service(service_name)
        self.send_query = rospy.ServiceProxy(service_name, bopt)
        self.iters = 0
        try:
            x_out, mvalue = self.optimizer.optimize()
            x_min = np.array([self.init_pose.position.x, self.init_pose.position.y, self.init_pose.position.z])
            x_min[self.regr_vars] = x_out
            res = self.send_query(init_pose, True, x_min, mvalue) # here the optimization has finished
            argmax_x_str = '[' + (', '.join([' {:.3f}'] * len(x_min))).format(*x_min).lstrip() + ']'
            rospy.loginfo('Minimum has been reached: ({:s}, {:.3f})'.format(argmax_x_str, mvalue))
        except rospy.ServiceException as e:
            rospy.logerr('The Service for bayesian optimization has been Terminated prematurely.')
            rospy.logerr('Terminating the node...')
            exit()
        # self.optimizer.plot_result1D()
        while not rospy.is_shutdown():
            rate.sleep()

    def init_messages(self, lb, ub, params):
        rospy.loginfo(rospy.get_name().split('/')[1] + ': Optimization bounds')
        rospy.loginfo(rospy.get_name().split('/')[1] + ': lower bounds {}'.format(lb.round(3)))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': upper bounds {}'.format(ub.round(3)))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': N iterations {:d}'.format(params.get('n_calls')))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': Acquisition function: {}'.format(params.get('acq_func')))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': noise in observations: {}'.format(params.get('noise')))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': xi: {}'.format(params.get('xi')))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': kappa: {}'.format(params.get('kappa')))
        
        if params.get('acq_func') == 'lbfgs':
            # The parameter make sense only in this case
            rospy.loginfo(rospy.get_name().split('/')[1] + ': N restarts {:d}'.format(params.get('n_restarts_optimizer')))

        if self.optimizer.deltaX is not None:
            rospy.loginfo(rospy.get_name().split('/')[1] + ': Stopping when queries are {:.2e} close'.format(self.optimizer.deltaX))
        if self.optimizer.deltaY is not None:
            rospy.loginfo(rospy.get_name().split('/')[1] + ': Stopping when best value is lower than {:.3f}'.format(self.optimizer.deltaY))
        if self.optimizer.checkpoint_file is not None:
            rospy.loginfo(rospy.get_name().split('/')[1] + ': saving checkpoints at {}'.format(self.optimizer.checkpoint_file))
    

    def min_function(self, Xin):
        p = deepcopy(self.init_pose)
        for idx in self.regr_vars:
            if idx == 0:
                p.position.x = float(Xin[idx])
            elif idx == 1:
                p.position.y = float(Xin[idx])
            elif idx == 2:
                p.position.z = float(Xin[idx])
        
        # p.position.x = float(Xin[0])
        # p.position.y = float(Xin[1])
        # p.position.z = float(Xin[2])
        rospy.loginfo('{:^10} {}'.format('ITERATION', self.iters))
        rospy.loginfo('Estimating metric at ({:.3f}, {:.3f}, {:.3f}) ...'.format(p.position.x, p.position.y, p.position.z))
        res = self.send_query(p, False, [], np.nan)
        rospy.loginfo('Estimatation of the metric obtained: {:.3f}'.format(res.Y))
        # res= boptResponse()
        self.iters += 1
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

    lb_x = [float(xx) for xx in rospy.get_param('~lb_x', [-.2, -.2, 0.])]
    ub_x = [float(xx) for xx in rospy.get_param('~ub_x', [.2, .2, 0.])]
    ee_link = rospy.get_param('~ee_link', 'hand_root')
    base_link = rospy.get_param('~base_link', 'world')
    service_name = rospy.get_param('~commander_service', 'bayes_optimization')
    n_iter = rospy.get_param('~bopt_iters', 20)
    chkpt_file = rospy.get_param('~checkpoint', None)

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

    params = {
        Skopt_BO.PARAMS.iters :n_iter,
        Skopt_BO.PARAMS.n_restarts_optimizer :1}
    n = len(lb_x)
    assert(len(lb_x) == len(ub_x))

    init_pos = np.array([
        current_pose.pose.position.x, 
        current_pose.pose.position.y,
        current_pose.pose.position.z])
    print(lb_x, init_pos)
    lb = init_pos[np.arange(len(lb_x))] + lb_x - 1e-10
    ub = init_pos[np.arange(len(ub_x))] + ub_x 
    
    BO_Node(n, params, lb=lb, ub=ub, init_pose=current_pose.pose, service_name=service_name, checkpoint=chkpt_file)
