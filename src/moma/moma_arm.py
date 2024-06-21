#!/usr/bin/env python

import os
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
from utils import conversions, math_utils
from . import moma_moveit_utils
import tf
import roslib
import pyassimp

roslib.load_manifest('gaussian_slam_msgs')  # Load your package manifest if necessary


def load_config(pkg_name, config_filename):
    pkg_path = roslib.packages.get_pkg_dir(pkg_name)
    config_path = os.path.join(pkg_path, 'config', config_filename)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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
      self.tf_listener = tf.TransformListener()
      print("MoMaArm configured")

      config = load_config('gaussian_slam_msgs', 'objects_config.yaml')
      moma_moveit_utils.add_objects_to_scene_yaml_topic(self.scene, config)

  def get_joint_states(self):
      joint_states = self.arm.get_current_joint_values()
      return joint_states
  
  def get_ik(self, pose):
      seed_state = self.get_joint_states()
      x, y, z, qx, qy, qz, qw  = conversions.pose_to_ik_format(pose)
      ik_result = self.ik_solver.get_ik(seed_state, x, y, z, qx, qy, qz, qw)

      if ik_result:
        rospy.loginfo("IK Result: %s", ik_result)
      else:
        rospy.logwarn("IK Solution not found")
      return ik_result

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

  def is_motion_plan_found(self, pose):
      self.arm.set_pose_target(pose)
      plan = self.arm.plan()
      #print("Plan: ", plan)
      success, trajectory, float_val, val_dict = plan
      #print("traj: ", trajectory)
      if success:  # If the list is not empty, a plan was found
          print("Reachable pose found ")
          return True
      return False
  
  def is_pose_reachable(self, pose):
      ik_result = self.get_ik(pose)
      if ik_result is not None:
        return True
      else:
        return False

  def is_within_fov(self, current_pose, new_position, fov, overlap_percentage, target_position):
    # Vector from current position to target position
    direction_to_target = [
        target_position[0] - current_pose.position.x,
        target_position[1] - current_pose.position.y,
        target_position[2] - current_pose.position.z
    ]

    # Vector from current position to new position
    direction_to_new_position = [
        new_position.x - current_pose.position.x,
        new_position.y - current_pose.position.y,
        new_position.z - current_pose.position.z
    ]

    # Calculate the angle between the target direction and the direction to the new position
    angle = math_utils.angle_between_vectors(direction_to_target, direction_to_new_position)

    # Convert FOV to radians
    fov_rad = math.radians(fov)
    half_fov = fov_rad / 2

    # Adjust the FOV bounds based on the overlap percentage
    overlap_fov = fov_rad * (overlap_percentage / 100)
    overlap_half_fov = overlap_fov / 2

    # Check if the angle to the new position falls within the overlap FOV bounds
    return -overlap_half_fov <= angle <= overlap_half_fov


  def generate_random_pose_within_fov(self,center, radius, z_min, z_max, current_pose, fov, overlap_percentage):
    pose = Pose()
    n = 0
    while n < 10:
        n = n+1
        pose.position.x = center[0] + radius * (random.random() - 0.5)
        pose.position.y = center[1] + radius * (random.random() - 0.5)
        pose.position.z = random.uniform(z_min, z_max)
        pose.orientation = self.calculate_orientation_towards_center(pose.position, center)
        if self.is_within_fov(current_pose, pose.position, fov, overlap_percentage, center):
            break
    return pose
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
      current_pose = self.get_current_pose()
      fov = 90
      for _ in range(num_poses):
          pose = self.generate_random_pose_within_fov(center, radius, z_min, z_max, current_pose, fov, 20)
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

  def get_current_pose(self):
    # Fetch the current pose of the robot from the tf listener
    try:
      (trans, rot) = self.tf_listener.lookupTransform('world', 'ur5etool0', rospy.Time(0))
      return conversions.trans_rot_to_pose(trans, rot)
      #return trans, rot
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      rospy.logwarn("Failed to get current robot pose")
      return None

  def generate_and_order_views(self,current_pose, center_position, num_views, fov, desired_fov_overlap):
    points = self.fibonacci_sphere_top_half(num_views, radius=1.0)

    views = []
    for point in points:
      pose = Pose()
      pose.position.x = point[0] + center_position[0]
      pose.position.y = point[1] + center_position[1]
      pose.position.z = point[2] + center_position[2]
      pose.orientation = conversions.direction_to_quaternion(point, center_position)
      views.append(pose)
