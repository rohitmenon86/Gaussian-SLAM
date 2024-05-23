#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import actionlib
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
from src.moma.moma_arm import MoMaArm
from src.moma.moma_base import MoMaBase
import tf

class MobileManipulator:
    def __init__(self, config_path):
        rospy.init_node('mobile_manipulator_node')
        self.arm = MoMaArm(config_path)
        self.base = MoMaBase()

    def perform_task(self):
        # Execute a sequence of movements based on the loaded configuration
        for name in self.arm.config['named_targets']:
            self.arm.move_to_named_target(name)
        for goal in self.arm.config['joint_goals']:
            self.arm.move_to_joint_target(goal)
        for pos in self.arm.config['cartesian_goals']:
            pose = Pose()
            pose.position.x = pos['position']['x']
            pose.position.y = pos['position']['y']
            pose.position.z = pos['position']['z']
            pose.orientation.w = pos['orientation']['w']
            if self.arm.is_pose_reachable(pose):
                self.arm.move_to_cartesian_target(pose)
            else:
                rospy.loginfo("Pose is not reachable.")

if __name__ == '__main__':
    config_path = ''
    moma = MobileManipulator(config_path)
    rospy.sleep(2)
    current_pose = moma.get_current_pose()
    if current_pose:
      current_position, current_orientation = current_pose
      x_offset = 2.0  # Distance to move in the x direction
      y_offset = 2.0  # Distance to move in the y direction
      theta_offset = 1.5708  # Turn 90 degrees

      new_x = current_position[0] + x_offset
      new_y = current_position[1] + y_offset
      current_yaw = tf.transformations.euler_from_quaternion(current_orientation)[2]
      new_theta = current_yaw + theta_offset

      # Move to the new goal
      #moma.move_to_goal(new_x, new_y, new_theta)
      moma.base.move_to_position(new_x, new_y, new_theta)