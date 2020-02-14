#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from bopt_grasp_quality.srv import bopt, boptResponse
from moveit_commander import MoveGroupCommander
from trac_ik_python.trac_ik import IK 

class Bopt_Commander_Mock():

    def __init__(self, fun, arm_group='ur5_arm'):
        rospy.init_node('bopt_metric_mock')
        self.ur5_commander = MoveGroupCommander(arm_group)
        self.IK = IK('world', 'hand_root', epsilon=1e-3, timeout=0.1, solve_type='Speed')
        self.metric_mock = fun
        self.service = rospy.Service('bayes_optimization', bopt, self.manage_query)
        self.query_hist = []
        self.reply_hist = []
        self.min_value = None
        rate = rospy.Rate(30)
        
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
        # rospy.sleep(.5)
        return boptResponse(Y)
        
    def min_function(self, query):
        target = PoseStamped()
        target.header.frame_id = 'world'
        target.header.stamp = rospy.Time().now()
        target.pose = query


        j_states = self.ur5_commander.get_current_joint_values()
        pos = [query.position.x, query.position.y, query.position.z]
        orient = [query.orientation.x, query.orientation.y, query.orientation.z, query.orientation.w]
        self.ur5_commander.set_pose_target(target)
        plan = self.ur5_commander.plan()
        self.ur5_commander.execute(plan, wait=True)
        # j_target = self.IK.get_ik(j_states, *pos + [0, 0, 0, 1])
        # self.ur5_commander.set_joint_value_target(j_target)
        # trajectory = self.ur5_commander.plan()
        # self.ur5_commander.execute(trajectory, wait=True)

        x = query.position.x
        Y = self.metric_mock(x) + np.random.normal(0.0, scale=.5) # the metric is noisy

        self.query_hist.append(x)
        self.reply_hist.append(Y)
        return Y


    def shutdown(self):

        rospy.loginfo('Closing Commander Node....')
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
        plt.plot(x, self.metric_mock(x), 'r')
        # plt.scatter(, )
        plt.draw()
        # plt.pause(.01)
        


if __name__ == "__main__":
    p = np.poly1d(np.poly([-5, -3, 0, 3]))
    fun = lambda x: p(x) + 2 * np.cos(2 * np.pi * 5 * x)
    x = np.arange(-10, 10, 0.01)
    
    Bopt_Commander_Mock(fun)
