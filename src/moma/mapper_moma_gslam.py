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
from src.moma.moma import MobileManipulator

import rospy
import rospkg
import roslib
import actionlib

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

class GSlamMapperMoMa():
  def __init__(self, gslam_config, moma_config) -> None:
    self.gslam_config = gslam_config
    self.moma_config = moma_config
    print(self.gslam_config["camera"]["manip_cam"])
    self.moma = MobileManipulator(moma_config)

    self.camera = Camera(self.gslam_config["camera"]["manip_cam"])
    self.data = CameraData(self.camera.intrinsics, self.camera.image_height, self.camera.image_width)
    self.gslam = GaussianSLAMROS(gslam_config, self.data)


  def update_camera_data(self):
    self.data.set_data(*self.camera.get_latest_data())
    self.data.intrinsics = self.camera.get_intrinsics()


def main(args):
    rospy.init_node('gslam_mapper_moma')
    rospack = rospkg.RosPack()
    gslam_pkg_path = rospack.get_path('gaussian_slam_ros')
    gslam_config_path = rospy.get_param("config_path", "configs/ros/realsense_ros.yaml")
    moma_config_path = rospy.get_param("moma_config_path", "configs/ros/moma.yaml")
    full_gslam_config_path = gslam_pkg_path + "/" +   gslam_config_path
    gslam_config = load_config(full_gslam_config_path)

    full_moma_config_path = gslam_pkg_path + "/" + moma_config_path
    moma_config = load_config(full_moma_config_path)

    setup_seed(gslam_config["seed"])
    rospy.loginfo("Configuring gslam_mapper_moma...")
    gslam_mapper_moma = GSlamMapperMoMa(gslam_config, moma_config)
    rospy.loginfo("...gslam_mapper_moma configured")
    rospy.spin()
