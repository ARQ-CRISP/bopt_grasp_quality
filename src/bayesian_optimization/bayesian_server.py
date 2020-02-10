#!/usr/bin/env python
from __future__ import print_function, division

import rospy

from bopt_grasp_quality.srv import bopt, Addpose, Addscore

def handle_pose(req):
    print("Returning [%s + %s = %s]"%(req.pose, score))
    return Addpose(req.pose + req.score)

def handle_score(req):
    print("Returning [%s + %s = %s]" % (req.pose, score))
    return Addscore(req.pose - req.score)

def add_two_ints_server():
    rospy.init_node('bayesian_server')
    s = rospy.Service('bayesian_pose', bopt, handle_pose)
    g = rospy.Service('bayesian_score', bopt, handle_score)  
    print("bayesian output is")
    rospy.spin()

if __name__ == "__main__":
    bayesian_server()

