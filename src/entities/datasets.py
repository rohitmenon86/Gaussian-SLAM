import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import json
import imageio

from src.entities.base_dataset import CameraData, BaseDataset
from src.entities.common_datasets import Replica, TUM_RGBD, ScanNet, ScanNetPP, RealSenseLive, ROSDatasetLive
from collections import deque

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
    elif dataset_name == "ros_live":
        return ROSDatasetLive
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
