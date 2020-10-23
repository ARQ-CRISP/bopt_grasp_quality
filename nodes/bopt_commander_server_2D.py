#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from bopt_grasp_quality.srv import bopt, boptResponse
from moveit_commander import MoveGroupCommander
from trac_ik_python.trac_ik import IK


class Bopt_Commander_Mock():

    def __init__(self, fun, arm_group='ur5_arm', noise_scale=.5):
        rospy.init_node('bopt_metric_mock')
        self.ur5_commander = MoveGroupCommander(arm_group)
        target = self.ur5_commander.get_named_target_values('start')
        self.ur5_commander.go(target)
        # self.IK = IK('world', 'hand_root', epsilon=1e-3, timeout=0.1, solve_type='Speed')
        self.metric_mock = fun
        self.service = rospy.Service(
            'bayes_optimization', bopt, self.manage_query)
        self.query_hist = []
        self.reply_hist = []
        self.min_value = None
        self.noise_scale = noise_scale
        rate = rospy.Rate(60)
        axis = [-10, 10, -80, 80]
        axis = [0, 1, -8, 8]
        fig = plt.figure()
        plt.ion()
        plt.axis(axis)
        ax = plt.axes(projection='3d')
        ax.set_title('Metric mock')
        ax.set_xlabel('EE x position')
        ax.set_ylabel('EE y position')
        ax.set_zlabel('Metric Value')
        # plt.axis([-1.5, 1.5, 0, 6])
        self.plot_fun2D(ax)
        axes = fig.axes
        plt.pause(.001)
        while not rospy.is_shutdown():
            plt.cla()
            fig.sca(*axes)
            # plt.axis(axis)

            ax.set_title('Metric mock')
            ax.set_xlabel('EE x position')
            ax.set_ylabel('EE y position')
            # fig.sca(fig.get_axes())
            self.plot_fun2D(ax)
            self.plot_datapoints2D(ax)
            ax.legend()
            plt.pause(.01)
            rate.sleep()

    def manage_query(self, query_msg):

        if not query_msg.foundmin:
            Y = self.min_function(query_msg.query)
        else:
            print("The minimum has been found",
                  (query_msg.minX, query_msg.minY))
            Y = np.nan
            self.min_value = (query_msg.minX, query_msg.minY)
        # rospy.sleep(.5)
        return boptResponse(Y)

    def min_function(self, query):
        target = PoseStamped()
        target.header.frame_id = 'world'
        target.header.stamp = rospy.Time().now()
        target.pose = query

        j_states = self.ur5_commander.get_current_joint_values()
        pos = [query.position.x, query.position.y, query.position.z]
        orient = [query.orientation.x, query.orientation.y,
                  query.orientation.z, query.orientation.w]
        self.ur5_commander.set_pose_target(target)
        plan = self.ur5_commander.plan()
        self.ur5_commander.execute(plan, wait=True)
        # j_target = self.IK.get_ik(j_states, *pos + [0, 0, 0, 1])
        # self.ur5_commander.set_joint_value_target(j_target)
        # trajectory = self.ur5_commander.plan()
        # self.ur5_commander.execute(trajectory, wait=True)

        X = [query.position.x, query.position.y, query.position.z]
        Y = self.metric_mock(np.array(X[0:2])) + np.random.normal(0.0,
                                                   scale=self.noise_scale)  # the metric is noisy

        self.query_hist.append(X)
        self.reply_hist.append(Y)
        return Y

    def shutdown(self):

        rospy.loginfo('Closing Commander Node....')
        rospy.sleep(.5)

    def plot_datapoints2D(self, ax):
        #TODO ADAPT to More Dimensions
        if len(self.query_hist) > 0:
            ax.scatter(np.array(self.query_hist)[:,0], np.array(self.query_hist)[:,1], self.reply_hist,
                        marker='x', label='sampled points', c='b', s=15, alpha=0.9)
            plt.draw()
            text = np.arange(1, len(self.query_hist)).astype(str).tolist()
            for xx, yy, tt in zip(self.query_hist, self.reply_hist, text):
                ax.text(xx[0] + .1, xx[1] + .1, yy + .1, tt, fontdict={'fontsize': 10},)

        if self.min_value is not None:
            Xmin, Ymin = self.min_value
            ax.scatter(Xmin[0], Xmin[1], Ymin, marker='o', label='minimum', c='r', s=25) 

    def plot_fun2D(self, ax):
        #TODO ADAPT to More Dimensions
        # plt.clf()
        boundaries = [[-1, 1], [-1, 1]]
        x_surf = np.linspace(
            (boundaries[0][0], boundaries[1][0]), 
            (boundaries[0][1], boundaries[1][1]), 1000)
        X_surf = np.dstack(np.meshgrid(x_surf[:,0], x_surf[:,1]))
        Z_surf = self.metric_mock(X_surf)
        surf = ax.plot_surface(X_surf[:,:,0], X_surf[:,:,1], Z_surf, cmap='viridis', edgecolor='none', alpha=0.2, label='metric function')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        # plt.plot(x, self.metric_mock(x), 'r', label='metric function')
        # plt.scatter(, )
        plt.draw()

        # plt.pause(.01)


if __name__ == "__main__":
    from scipy.stats import norm, multivariate_normal

    g0 = multivariate_normal(np.r_[-2, 0], 0.5*np.eye(2))
    g1 = multivariate_normal(np.r_[1., 1.], 0.1*np.eye(2))
    g2 = multivariate_normal(np.r_[-2, -2.], 0.2*np.eye(2))
    p = lambda x: (x**2).sum(axis=-1)

    fun = lambda x: 40 * g0.pdf(x) - 10*g1.pdf(x) - 10*g2.pdf(x) + p(x)


    def fun(x): 
        # x[:,0] = (x[:,0] - 1) * 5
        y = x * 2. +0.5
        return 40 * g0.pdf(y) - 10*g1.pdf(y) - 10*g2.pdf(y) + p(y)
    # x = np.arange(-10, 10, 0.01)

    Bopt_Commander_Mock(fun, noise_scale=.1)
