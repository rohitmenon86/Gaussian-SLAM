import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import json
import imageio

import base_dataset
import common_datasets
from common_datasets import Replica, TUM_RGBD, ScanNet, ScanNetPP, RealSenseLive
from collections import deque

class CameraData():
    def __init__(self, rgb_intrinsics, height, width) -> None:
        self.intrinsics = rgb_intrinsics
        self.height = height
        self.width  = width
        self.rgb_array = None
        self.depth_array = None
        self.c2w = None
        self.timestamp = None

        self.rgb_images = deque(maxlen=3)
        self.depth_images = deque(maxlen=3)
        self.c2ws = deque(maxlen=3)
        self.timestamps = deque(maxlen=3)
    def set_data(self, rgb, depth, c2w, timestamp):
        self.rgb_array = rgb
        self.depth_array = depth
        self.c2w = c2w
        self.timestamp = timestamp
    
    def __getitem__(self, index):
        return index, self.rgb_array, self.depth_array, self.c2w


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == "realsense":
        return RealSenseLive
    elif dataset_name == "ros":
        return ROS
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
