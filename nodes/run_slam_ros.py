#!/home/rohit/workspace/software/anaconda3/envs/gslam/bin/python
import argparse
import os
import sys
import time
import uuid

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
class GaussianSlamROSNode():
    def __init__(self, config) -> None:
        self.config = config
        print(self.config["camera"]["manip_cam"])
        self.camera = Camera(self.config["camera"]["manip_cam"])
        self.timer = None #rospy.Timer(rospy.Duration(0.5), self.process_data)

         # Create the action servers
        self.start_server  = actionlib.SimpleActionServer('~start', StartAction, self.start_cb, False)
        self.stop_server   = actionlib.SimpleActionServer('~stop', StopAction, self.stop_cb, False)
        
        self.pause_server  = actionlib.SimpleActionServer('~pause', PauseAction, self.pause_cb, False)
        self.resume_server = actionlib.SimpleActionServer('~resume', ResumeAction, self.resume_cb, False)
        self.evaluate_server = actionlib.SimpleActionServer('~evaluate', EvaluateAction, self.evaluate_cb, False)


        self.data = CameraData(self.camera.intrinsics, self.camera.image_height, self.camera.image_width)
        self.gslam = GaussianSLAMROS(config, self.data)
        #self.evaluator = Evaluator(self.gslam.output_path, self.gslam.output_path / "config.yaml")
        # Start action servers
        self.start_server.start()
        self.pause_server.start()
        self.resume_server.start()
        self.stop_server.start()
        self.evaluate_server.start()
        
    def update_camera_data(self):
        self.data.set_data(*self.camera.get_latest_data())
        self.data.intrinsics = self.camera.get_intrinsics()
        
    def process_data(self, event):
        """
        Callback method for the timer event.
        `event` contains the last time the callback was called and the current time it was called.
        """
        if self.is_paused:
            rospy.logwarn_throttle(10.0, "Processing data paused")
            return
        if self.is_stopped:
            rospy.logwarn_throttle(10.0, "Processing data stopped")
            return
        if self.camera.is_valid_data_available == False:
            rospy.logwarn("No valid data available. Hence not processing gaussian slam")
            return
        rospy.loginfo_throttle(1.0, "Getting camera data")
        self.update_camera_data()
        rospy.loginfo_throttle(1.0, "Processing latest camera data")
        self.gslam.process(self.data)

    def start_cb(self, goal):
        if self.timer is not None:
            self.timer.shutdown()
        self.timer = rospy.Timer(rospy.Duration(0.5), self.process_data)
        self.is_paused = False
        self.is_stopped = False
        self.start_server.set_succeeded()

    def pause_cb(self, goal):
        self.is_paused = True
        self.pause_server.set_succeeded()

    def resume_cb(self, goal):
        self.is_paused = False
        self.resume_server.set_succeeded()

    def stop_cb(self, goal):
        self.is_stopped = True
        # Save the data to a file before shutting down the timer
        self.gslam.save_dataset_and_ckpt()
        
        # Shutdown the timer after saving the data
        # if self.timer is not None:
        #     self.timer.shutdown()
        #     self.timer = None
        
        self.stop_server.set_succeeded()

    def evaluate_cb(self, goal):
        self.is_stopped = True
        # Save the data to a file before shutting down the timer
        self.gslam.save_dataset_and_ckpt()

        if self.evaluate_server.is_preempt_requested():
            self.evaluate_server.set_preempted()
        else:
            # Proceed with evaluating the data
            result = EvaluateResult()
            self.evaluator = Evaluator(self.gslam.output_path, self.gslam.output_path / "config.yaml")
            self.evaluator.run()
            result.evaluation = "Data evaluated successfully with {} data points.".format(len(self.dataset))
            self.evaluate_server.set_succeeded(result)
            rospy.loginfo("Evaluation completed.")

def main(args):
    rospy.init_node('gaussian_slam_ros')
    rospack = rospkg.RosPack()
    gslam_path = rospack.get_path('gaussian_slam_ros')
    config_path = rospy.get_param("config_path", "configs/ros/realsense_ros.yaml")
    full_config_path = gslam_path + "/" +   config_path
    config = load_config(full_config_path)
    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir="/home/rohit/workspace/scratch/outputs/slam/wandb",
            group=config["data"]["scene_name"],
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    setup_seed(config["seed"])
    rospy.loginfo("Configuring GaussianSlamROSNode...")
    gslam_ros_node = GaussianSlamROSNode(config)
    rospy.loginfo("...GaussianSlamROSNode configured")
    rospy.spin()
    # gslam_ros_node.evaluator.run()
    # if config["use_wandb"]:
    #     evals = ["rendering_metrics.json",
    #              "reconstruction_metrics.json", "ate_aligned.json"]
    #     log_metrics_to_wandb(evals, gslam_ros_node.gslam.output_path, "Evaluation")
    #     wandb.finish()
    print("All done.✨")

if __name__ == "__main__":
    print("Starting")
    main(sys.argv)
