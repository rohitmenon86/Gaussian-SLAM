#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import actionlib
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos

class MoMaArm:
  def __init__(self, config_path = ""):
      # Load configuration
    #   with open(config_path, 'r') as file:
    #       self.config = yaml.safe_load(file)
      
      moveit_commander.roscpp_initialize(sys.argv)
      self.robot = moveit_commander.RobotCommander()
      self.scene = moveit_commander.PlanningSceneInterface()
      #self.arm = moveit_commander.MoveGroupCommander(self.config['arm_group_name'])
      self.arm = moveit_commander.MoveGroupCommander("manipulator")
      print("MoMaArm configured")

  def move_to_joint_target(self, joint_goals):
      self.arm.go(joint_goals, wait=True)
      self.arm.stop()

  def move_to_named_target(self, name):
      self.arm.set_named_target(name)
      self.arm.go(wait=True)
      self.arm.stop()

  def move_to_cartesian_target(self, pose):
      waypoints = [pose]
      (plan, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, 0.0)
      self.arm.execute(plan, wait=True)

  def is_pose_reachable(self, pose):
      self.arm.set_pose_target(pose)
      plan = self.arm.plan()
      if plan.joint_trajectory.points:  # If the list is not empty, a plan was found
          return True
      return False
  
  

