import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import json
import imageio
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
        self.pose = np.eye(4)

        self.rgb_images = deque(maxlen=3)
        self.depth_images = deque(maxlen=3)
        self.c2ws = deque(maxlen=3)
        self.timestamps = deque(maxlen=3)
    def set_data(self, rgb, depth, c2w, timestamp):
        self.rgb_array = rgb
        self.depth_array = depth
        self.c2w = c2w
        self.timestamp = timestamp
        self.pose = c2w #np.eye(4)
    
    def __getitem__(self, index):
        return index, self.rgb_array, self.depth_array, self.c2w

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)