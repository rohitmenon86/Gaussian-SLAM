#!/usr/bin/env python

import sys
import os
import rospy
import moveit_commander
import actionlib
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
import numpy as np

script_dir = os.path.dirname(__file__)  # Gets the directory where the script is running
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory (project directory)
src_dir = os.path.join(parent_dir, 'src')  # Constructs the path to the 'src' folder

sys.path.append(parent_dir)
# Add 'src' directory to the Python path
sys.path.append(src_dir)
from src.moma.moma_arm import MoMaArm
from src.moma.moma_base import MoMaBase
from src.moma.moma  import MobileManipulator

import tf

if __name__ == '__main__':
  rospy.init_node('mobile_manipulator_goal_navigation')
  use_border_goals = rospy.get_param('~use_border_goals', True)
  manipulator_base = MoMaBase(use_border_goals)

  if not use_border_goals:
    # Set external goals only if the external strategy is used
    x = rospy.get_param('~goal_x', 5.0)
    y = rospy.get_param('~goal_y', 5.0)
    theta = rospy.get_param('~goal_theta', np.pi / 2)
    manipulator_base.set_external_goal(x, y, theta)

  manipulator_base.autonomous_exploration()