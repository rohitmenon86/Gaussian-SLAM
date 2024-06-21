#!/home/rohit/workspace/software/anaconda3/envs/gslam/bin/python
import sys
import os
import time
import uuid
import argparse
import yaml



script_dir = os.path.dirname(__file__)  # Gets the directory where the script is running
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory (project directory)
pp_dir = os.path.dirname(parent_dir)  # Gets the parent directory (project directory)
src_dir = os.path.join(parent_dir, 'src')  # Constructs the path to the 'src' folder

sys.path.append(pp_dir)
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
from utils import conversions, math_utils

class GSlamMapperMoMa():
  def __init__(self, gslam_config, moma_config=None) -> None:
    self.gslam_config = gslam_config
    self.moma_config = moma_config
    print(self.gslam_config["camera"]["manip_cam"])
    self.moma_arm = MoMaArm()

    self.camera = Camera(self.gslam_config["camera"]["manip_cam"])
    self.data = CameraData(self.camera.intrinsics, self.camera.image_height, self.camera.image_width)
    self.gslam = GaussianSLAMROS(gslam_config, self.data)

    self.start_server  = actionlib.SimpleActionServer('~start', StartAction, self.render_viewpoints, False)
    
    self.moma_arm.move_to_named_target("obs_pose_3")
    self.train_gslam_on_current_camera_data()

    self.moma_arm.move_to_named_target("obs_pose_1")
    self.train_gslam_on_current_camera_data()

    self.moma_arm.move_to_named_target("obs_pose_2")
    self.train_gslam_on_current_camera_data()


    self.start_server.start()

  def update_camera_data(self):
    self.data.set_data(*self.camera.get_latest_data())
    self.data.intrinsics = self.camera.get_intrinsics()

  def train_gslam_on_current_camera_data(self):
    self.update_camera_data()
    self.gslam.process(self.data, True)

  def render_viewpoints(self, goal):
    reachable_poses = self.moma_arm.generate_random_reachable_poses(center=[0.8, 0.0, 0.6], radius=0.5, num_poses=10, z_min=0.8, z_max=1.2)

    i = 2
    for pose in reachable_poses.poses:
      c2w_np = conversions.pose_to_matrix(pose)
      render_data = self.gslam.render_and_save(c2w_np, i)
      self.moma_arm.move_to_cartesian_target(pose)
      self.train_gslam_on_current_camera_data()
      i = i + 1


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

def generate_and_order_views(current_pose, center_position, num_views, fov, desired_fov_overlap, min_distance, min_z, max_z):
    points = math_utils.fibonacci_sphere_top_half(num_views, radius=1.0, min_distance=min_distance, min_z=min_z, max_z=max_z)

    views = []
    for point in points:
        pose = Pose()
        pose.position.x = point[0] + center_position[0]
        pose.position.y = point[1] + center_position[1]
        pose.position.z = point[2] + center_position[2]
        pose.orientation = conversions.direction_to_quaternion(point, center_position)
        views.append(pose)

    # Calculate distance from current pose to each generated pose
    def distance_from_current(pose):
        return np.sqrt((pose.position.x - current_pose.position.x)**2 + 
                       (pose.position.y - current_pose.position.y)**2 + 
                       (pose.position.z - current_pose.position.z)**2)

    views.sort(key=distance_from_current)
    return views

def main(args):
    rospy.init_node('mapper_moma_gslam')
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

if __name__ == "__main__":
    print("Starting")
    main(sys.argv)