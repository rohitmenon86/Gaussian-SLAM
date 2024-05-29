#!/home/rohit/workspace/software/anaconda3/envs/gslam/bin/python
import sys
import os
import time
import uuid
import argparse
import yaml



script_dir = os.path.dirname(__file__)  # Gets the directory where the script is running
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory (project directory)
src_dir = os.path.join(parent_dir, 'src')  # Constructs the path to the 'src' folder

sys.path.append(parent_dir)
# Add 'src' directory to the Python path
sys.path.append(src_dir)

import wandb

from src.entities.gaussian_slam import GaussianSLAM
from src.entities.gaussian_slam_ros import GaussianSLAMROS
from src.entities.base_dataset import CameraData
from src.entities.camera_ros import Camera
from src.evaluation.evaluator import Evaluator
from src.utils.io_utils import load_config, log_metrics_to_wandb
from src.utils.utils import setup_seed
from src.moma.moma import MobileManipulator, MoMaArm

import rospy
import rospkg
import roslib
import actionlib
import math

import message_filters
import tf2_ros
import gaussian_slam_msgs
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from gaussian_slam_msgs.msg import StartAction, PauseAction, ResumeAction, StopAction, EvaluateAction, EvaluateResult
from threading import Lock
import tf.transformations as transformations
import numpy as np
from collections import deque
from geometry_msgs.msg import Pose, PoseStamped
from math import sqrt, atan2, radians, cos, sin

class GSlamMapperMoMa():
  def __init__(self, gslam_config, moma_config=None) -> None:
    self.gslam_config = gslam_config
    self.moma_config = moma_config
    print(self.gslam_config["camera"]["manip_cam"])
    self.moma_arm = MoMaArm()

    self.camera = Camera(self.gslam_config["camera"]["manip_cam"])
    self.data = CameraData(self.camera.intrinsics, self.camera.image_height, self.camera.image_width)
    self.gslam = GaussianSLAMROS(gslam_config, self.data)


  def update_camera_data(self):
    self.data.set_data(*self.camera.get_latest_data())
    self.data.intrinsics = self.camera.get_intrinsics()

  def train_gslam_on_current_camera_data(self):
    self.update_camera_data()
    self.gslam.process(self.data, True)

  def render_viewpoints(self):
     self.moma_arm.generate_random_reachable_poses()


  def generate_camera_positions(self, n, radius, overlap):
    """
    Generate camera positions on a spherical dome over a table to ensure complete coverage.

    :param n: Number of camera positions to generate
    :param radius: Radius of the imaginary sphere centered over the table
    :param overlap: Desired overlap in degrees between the views of adjacent cameras
    """
    fov_rad = np.deg2rad(self.camera.fov_horizontal)
    overlap_rad = np.deg2rad(overlap)
    angular_distance = fov_rad - overlap_rad

    viewpoints = []
    for i in range(n):
      phi = np.arccos(1 - 2 * (i + 0.5) / n)
      theta = angular_distance * (i + 0.5)
      
      x = radius * np.sin(phi) * np.cos(theta)
      y = radius * np.sin(phi) * np.sin(theta)
      z = radius * np.cos(phi) + self.table_height  # Adjust z to be above the table

      # Create a pose to check reachability
      pose = Pose()
      pose.position.x = x
      pose.position.y = y
      pose.position.z = z
      pose.orientation.w = 1.0  # Assuming the camera always faces downward

      # Check if the pose is reachable and valid
      if self.mobile_manipulator.arm.is_pose_reachable(pose):
        viewpoints.append((x, y, z))
      else:
        rospy.loginfo("Pose at ({}, {}, {}) is not reachable.".format(x, y, z))

    return viewpoints

  def find_viewpoints(self):
      """
      High-level function to find and return all valid viewpoints over a table.
      """
      n_cameras = 10       # Number of cameras
      radius = 1           # Radius of the sphere
      desired_overlap = 10 # Desired overlap in degrees between the views of adjacent cameras

      return self.generate_camera_positions(n_cameras, radius, desired_overlap)

  def calculate_new_pose(self, current_pose, object_pose, theta_offset):
        # Calculate the angle to the object
        dx = object_pose.x - current_pose.x
        dy = object_pose.y - current_pose.y
        angle_to_object = atan2(dy, dx)

        # Apply the theta offset
        new_angle = angle_to_object + radians(theta_offset)

        # Calculate new position that is theta_offset degrees rotated around the object
        distance = sqrt(dx**2 + dy**2)
        new_x = object_pose.x - distance * cos(new_angle)
        new_y = object_pose.y - distance * sin(new_angle)

        # Calculate the orientation to still point towards the object
        orientation_to_object = atan2(-sin(new_angle), -cos(new_angle))  # Reverse angle to face the object

        return new_x, new_y, orientation_to_object

def main(args):
    rospy.init_node('gslam_mapper_moma')
    rospack = rospkg.RosPack()
    gslam_pkg_path = rospack.get_path('gaussian_slam_ros')
    gslam_config_path = rospy.get_param("config_path", "configs/ros/realsense_ros.yaml")
    full_gslam_config_path = gslam_pkg_path + "/" +   gslam_config_path
    gslam_config = load_config(full_gslam_config_path)

    #moma_config_path = rospy.get_param("moma_config_path", "configs/ros/moma.yaml")
    #full_moma_config_path = gslam_pkg_path + "/" + moma_config_path
    #moma_config = load_config(full_moma_config_path)

    setup_seed(gslam_config["seed"])
    rospy.loginfo("Configuring gslam_mapper_moma...")
    gslam_mapper_moma = GSlamMapperMoMa(gslam_config)
    rospy.loginfo("...gslam_mapper_moma configured")
    rospy.spin()
