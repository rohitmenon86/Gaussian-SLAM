#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import actionlib
import yaml
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, PoseArray, Point
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
import random
from trac_ik_python.trac_ik import IK
from utils import conversions

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
      self.pose_pub = rospy.Publisher('/random_poses', PoseArray, queue_size=10)
      self.center_pub = rospy.Publisher('/center_pose', PoseStamped, queue_size=10)
      self.ik_solver = IK("world", "ur5etool0")
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
      #print("Plan: ", plan)
      success, trajectory, float_val, val_dict = plan
      #print("traj: ", trajectory)
      if success:  # If the list is not empty, a plan was found
          print("Reachable pose found ")
          return True
      return False
  
  def generate_random_pose(self, center, radius, z_min, z_max):
      pose = Pose()
      pose.position.x = center[0] + radius * (random.random() - 0.5)
      pose.position.y = center[1] + radius * (random.random() - 0.5)
      pose.position.z = random.uniform(z_min, z_max)
      pose.orientation = self.calculate_orientation_towards_center(pose.position, center)
      return pose
  
  def generate_random_reachable_poses(self, center, radius, num_poses, z_min, z_max):
      reachable_poses = PoseArray()
      reachable_poses.header = Header()
      reachable_poses.header.frame_id = "world"
      reachable_poses.header.stamp = rospy.Time.now()
      for _ in range(num_poses):
          pose = self.generate_random_pose(center, radius, z_min, z_max)
          print("Random Pose: ", pose)
          if self.is_pose_reachable(pose):
              reachable_poses.poses.append(pose)
              print(f"Reachable Pose {len(reachable_poses.poses)}: {pose}")

      self.pose_pub.publish(reachable_poses)

      center_pose = PoseStamped()
      center_pose.header = reachable_poses.header
      center_pose.pose.position = Point(center[0], center[1], center[2] if len(center) > 2 else 0)
      center_pose.pose.orientation = Quaternion(0, 0, 0, 1)  # Default orientation
      self.center_pub.publish(center_pose)
      
      return reachable_poses
  
  def calculate_orientation_towards_center(self, position, center):
      # Calculate the direction vector
      dx = position.x - center[0]
      dy = position.y - center[1]
      dz = position.z - center[2]  if len(center) > 2 else 0

      # Normalize the direction vector
      length = math.sqrt(dx**2 + dy**2 + dz**2)
      if length == 0:
          return Quaternion(0, 0, 0, 1)  # No orientation change if direction vector is zero

      dx /= length
      dy /= length
      dz /= length

      # Calculate yaw and pitch to point towards the center
      yaw = math.atan2(dy, dx)
      pitch = math.atan2(dz, math.sqrt(dx**2 + dy**2))
      roll = 0  # Assuming no roll

      return conversions.euler_to_quaternion(roll, pitch, yaw)


  
