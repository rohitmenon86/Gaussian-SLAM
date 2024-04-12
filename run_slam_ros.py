import argparse
import os
import sys
import time
import uuid

import wandb

from src.entities.gaussian_slam import GaussianSLAM
from src.entities.gaussian_slam_ros import GaussianSLAMROS

from src.evaluation.evaluator import Evaluator
from src.utils.io_utils import load_config, log_metrics_to_wandb
from src.utils.utils import setup_seed

import rospy
import rospkg
import roslib

import message_filters
import tf2_ros
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from threading import Lock
import tf.transformations as transformations
import numpy as np
from collections import deque

class Camera():
    def __init__(self, camera_config) -> None:
        self.camera_config = camera_config
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber(self.camera_config["rgb_topic"], Image)
        self.depth_sub = rospy.Subscriber(self.camera_config["depth_topic"], Image)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        # ApproximateTimeSynchronizer is used to approximate synchronization of the topics
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)
        
        rgb_info = rospy.wait_for_message(self.camera_config["rgb_info_topic"], CameraInfo)
        self.intrinsics = np.array(rgb_info.K).reshape((3, 3))
        self.image_height = rgb_info.height
        self.image_width  = rgb_info.width

        self.rgb_info_sub = rospy.Subscriber(self.camera_config["rgb_info_topic"], CameraInfo, self.rgb_info_callback, queue_size=1)
        self.depth_info_sub = rospy.Subscriber(self.camera_config["depth_info_topic"], CameraInfo, self.depth_info_callback, queue_size=1)
        # Storage for the most recent synchronized messages
        self.rgb_image = None
        self.depth_image = None
        self.camera_transform = None

        self.lock = Lock()
    
    def rgb_info_callback(self, camera_info):
        with self.lock:
            self.intrinsics = np.array(camera_info.K).reshape((3, 3))

    def get_intrinsics(self):
        intrinsics = None
        with self.lock:
            intrinsics = self.intrinsics
        return intrinsics
    
    def depth_info_callback(self, camera_info):
        with self.lock:
            self.depth_intrinsics = np.array(camera_info.K).reshape((3, 3))

    def get_depth_intrinsics(self):
        intrinsics = None
        with self.lock:
            intrinsics = self.depth_intrinsics
        return intrinsics
    
    def callback(self, rgb_image, depth_image):
        try:
            # Try to get the transform when the images arrive
            camera_transform = self.tf_buffer.lookup_transform(
                'world', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
            with self.lock:
                self.rgb_image = rgb_image
                self.depth_image = depth_image
                self.camera_transform = camera_transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to get camera transform: %s", e)

    def transform_to_matrix(self, transform):
        """ Convert a Transform message into a 4x4 numpy transformation matrix. """
        trans = [transform.translation.x, transform.translation.y, transform.translation.z]
        rot = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        translation_matrix = transformations.translation_matrix(trans)
        rotation_matrix = transformations.quaternion_matrix(rot)
        return transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    def get_latest_data(self):
        with self.lock:
            try:
                rgb_data = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
                depth_data = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
                pose = self.transform_to_matrix(self.camera_transform.transform)
                timestamp = self.rgb_image.header.stamp
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: %s", e)
                return None, None, None, None
            return rgb_data, depth_data, pose, timestamp
        
class GaussianSlamROSNode():
    def __init__(self, config) -> None:
        self.config = config
        self.camera = Camera(self.config["camera"]["camera_name"])
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)

        self.data = CameraData(self.camera.intrinsics, self.camera.image_height, self.camera.image_width)

        self.gslam = GaussianSLAMROS(config)
        self.evaluator = Evaluator(self.gslam.output_path, self.gslam.output_path / "config.yaml")

    def update_camera_data(self):
        self.data.set_data(*self.camera.get_latest_data())
        self.data.intrinsics = self.camera.get_intrinsics

    def timer_callback(self, event):
        """
        Callback method for the timer event.
        `event` contains the last time the callback was called and the current time it was called.
        """
        self.update_camera_data()
        self.gslam.process(self.data)

def main(args):
    rospy.init_node('gaussian_slam_ros')
    rospack = rospkg.RosPack()
    gslam_path = rospack.get_path('yolo_segmentation_ros')
    config_path = rospy.get_param("config_path", "configs/ros/realsense_ros.yaml")
    full_config_path = gslam_path + config_path
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
    gslam_ros_node = GaussianSlamROSNode(config)
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
