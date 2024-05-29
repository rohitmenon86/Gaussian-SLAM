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
  rospy.init_node('moma_arm_test')
  use_border_goals = rospy.get_param('~use_border_goals', True)
  moma_arm = MoMaArm()
  moma_arm.generate_random_reachable_poses([0.72, 0.0, 0.61], 0.4, 10, 0.8, 1.2)