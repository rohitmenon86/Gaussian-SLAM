import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_matrix
import math

def pose_to_matrix(pose: Pose) -> np.ndarray:
    # Extract the position (translation)
    tx = pose.position.x
    ty = pose.position.y
    tz = pose.position.z

    # Extract the orientation (quaternion)
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w

    # Create a rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix([qx, qy, qz, qw])

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    
    return Quaternion(qx, qy, qz, qw)