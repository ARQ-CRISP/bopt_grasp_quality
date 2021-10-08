#!/usr/bin/env python
from __future__ import division, print_function

import numpy as np
import rospy
from rospkg.rospack import RosPack
from copy import deepcopy
from tf2_ros import TransformListener, Buffer
from bayesian_optimization import Skopt_BO, Skopt_UBO
from bayesian_optimization import Random_Explorer
from bopt_grasp_quality.srv import bopt, boptResponse
# from math import nan
from geometry_msgs.msg import PoseStamped, Pose, Transform


class BO_Node(object):

    __package_name = 'bopt_grasp_quality'
    def __init__(
        self, n, params, lb=None, ub=None, init_pose = Pose(),
        service_name='bayes_optimization', checkpoint='BayesOpt.pkl'):
        
        lb, ub, params = self._process_args(
            service_name, init_pose, n, ub, lb, params, checkpoint)
        
        self.optimizer = Skopt_BO(
            len(self.regr_vars), self.min_function, params,
            lb=lb[self.regr_vars], ub=ub[self.regr_vars])
        
        self.init_messages(lb, ub, params)
        # self.optimizer.set_Xstopping_callback(1e-3)
        if checkpoint is not None:
            self.optimizer.set_checkpointing(self.checkpoint_file)
            
        
        
    def _process_args(self, service_name, init_pose, n, ub, lb, params, checkpoint):
        pack_path = RosPack().get_path(self.__package_name)
        self.service_name = service_name
        self.init_pose = init_pose
        self.checkpoint_file = None if checkpoint is None else pack_path + '/etc/' + checkpoint
        rospy.on_shutdown(self.shutdown)
        if ub is not None or lb is not None:
            self.regr_vars = np.where((np.array(ub) - np.array(lb)) >= 1e-6)[0]
        else:
            self.regr_vars = np.arange(n)
        
        return lb, ub, params
    
    def node_run(self):
        rate = rospy.Rate(10)
        self.iters = 0
        rospy.wait_for_service(self.service_name)
        self.send_query = rospy.ServiceProxy(self.service_name, bopt)
        
        try:
            x_out, mvalue = self.optimizer.optimize()
            x_min = np.array([self.init_pose.position.x, self.init_pose.position.y, self.init_pose.position.z])
            x_min[self.regr_vars] = x_out
            res = self.send_query(self.init_pose, True, x_min, mvalue) # here the optimization has finished
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
        init = np.array([
            self.init_pose.position.x,
            self.init_pose.position.y,
            self.init_pose.position.z
        ])
        init[self.regr_vars] = Xin
        for i, v in enumerate(init):
            if i == 0:
                p.position.x = float(v)
            elif i == 1:
                p.position.y = float(v)
            elif i == 2:
                p.position.z = float(v)
                
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

class UBO_Node(BO_Node):
    
    def __init__(self, n, params, lb=None, ub=None, init_pose = Pose(),
                 ut_cov=0.001, sigma_params={'alpha': .3, 'beta': 2., 'kappa': .1},
                 service_name='bayes_optimization', checkpoint='UnscentedBayesOpt.pkl'):
        
    
        lb, ub, params = self._process_args(
            service_name, init_pose, n, ub, lb, params, checkpoint)
        
        self.optimizer = Skopt_UBO(
            len(self.regr_vars), self.min_function, ut_cov, sigma_params, params,
            lb=lb[self.regr_vars], ub=ub[self.regr_vars])
        # self.optimizer.set_Xstopping_callback(1e-3)
        self.init_messages(lb, ub, params)
        if checkpoint is not None:
            self.optimizer.set_checkpointing(self.checkpoint_file)
             

class RS_Node():

    __package_name = 'bopt_grasp_quality'
    def __init__(self, n, params, lb=None, ub=None, init_pose = Pose(), service_name='bayes_optimization', checkpoint='RanSrch.pkl'):
        
        rospy.on_shutdown(self.shutdown)
        pack_path = RosPack().get_path(self.__package_name)
        rate = rospy.Rate(30)
        self.init_pose = init_pose
        if ub is not None or lb is not None:
            self.regr_vars = np.where((np.array(ub) - np.array(lb)) >= 1e-6)[0]
        else:
            self.regr_vars = np.arange(n)
        if Random_Explorer.PARAMS.init_pos in params and params[Random_Explorer.PARAMS.init_pos] is not None:
            params[Random_Explorer.PARAMS.init_pos] = np.array(params[Random_Explorer.PARAMS.init_pos])[self.regr_vars]
        self.optimizer = Random_Explorer(
            len(self.regr_vars), self.min_function, 
            lb=lb[self.regr_vars], ub=ub[self.regr_vars], params=params)
        self.optimizer.set_checkpointing(pack_path + '/etc/' + checkpoint)
        # self.optimizer = Random_Explorer(n, self.min_function, params, lb=lb, ub=ub)
        # self.optimizer.set_Xstopping_callback(1e-3)
        # self.optimizer.set_checkpointing(pack_path + '/etc/' + checkpoint)

        self.init_messages(lb, ub, params)
        rospy.wait_for_service(service_name)
        self.send_query = rospy.ServiceProxy(service_name, bopt)
        self.iters = 0
        try:
            x_out, mvalue = self.optimizer.optimize()
            # x_out = x_out[0]

            x_min = np.array([self.init_pose.position.x, self.init_pose.position.y, self.init_pose.position.z])
            x_min[self.regr_vars] = np.array(x_out)
            res = self.send_query(init_pose, True, x_min, mvalue) # here the optimization has finished
            argmax_x_str = '[' + (', '.join([' {:.3f}'] * len(x_min))).format(*x_min).lstrip() + ']'
            rospy.loginfo('Minimum has been reached: ({:s}, {:.3f})'.format(argmax_x_str, mvalue))
            # rospy.loginfo('Minimum has been reached: ({:.3f}, {:.3f})'.format(x_out, mvalue))
            
        except rospy.ServiceException as e:
            rospy.logerr('The Service for bayesian optimization has been Terminated prematurely.')
            rospy.logerr('Terminating the node...')
            exit()
        # self.optimizer.plot_result1D()
        while not rospy.is_shutdown():
            rate.sleep()

    def init_messages(self, lb, ub, params):
        rospy.loginfo(rospy.get_name().split('/')[1] + ': Optimization bounds')
        rospy.loginfo(rospy.get_name().split('/')[1] + ': - lower bounds {}'.format(lb.round(3)))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': - upper bounds {}'.format(ub.round(3)))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': N iterations {:d}'.format(params.get('n_calls')))
        if params.get('init') is not None:
            rospy.loginfo(rospy.get_name().split('/')[1] + ': Initialization at: {}'.format(params.get('init')))
        rospy.loginfo(rospy.get_name().split('/')[1] + ': discretizations values: {}'.format(params.get('diff')))    

        # if self.optimizer.deltaX is not None:
            # rospy.loginfo(rospy.get_name().split('/')[1] + ': Stopping when queries are {:.2e} close'.format(self.optimizer.deltaX))
        # if self.optimizer.deltaY is not None:
            # rospy.loginfo(rospy.get_name().split('/')[1] + ': Stopping when best value is lower than {:.3f}'.format(self.optimizer.deltaY))
        # if self.optimizer.checkpoint_file is not None:
            # rospy.loginfo(rospy.get_name().split('/')[1] + ': saving checkpoints at {}'.format(self.optimizer.checkpoint_file))
    

    def min_function(self, Xin):
        p = deepcopy(self.init_pose)
        init = np.array([
            self.init_pose.position.x,
            self.init_pose.position.y,
            self.init_pose.position.z
        ])
        init[self.regr_vars] = Xin
        for i, v in enumerate(init):
            if i == 0:
                p.position.x = float(v)
            elif i == 1:
                p.position.y = float(v)
            elif i == 2:
                p.position.z = float(v)
        # p.position.y = float(Xin)
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
