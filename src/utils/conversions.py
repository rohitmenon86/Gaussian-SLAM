import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Point
from tf.transformations import quaternion_matrix
import math
from scipy.spatial.transform import Rotation as R


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

def trans_rot_to_pose(trans, rot):
    pose = Pose()
    pose.position = Point(*trans)
    pose.orientation = Quaternion(*rot)
    return pose

def pose_to_ik_format(pose):
    # Extract position
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z

    # Extract orientation
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w

    return x, y, z, qx, qy, qz, qw

def direction_to_quaternion(direction, center):
    direction = np.array(direction) - np.array(center)
    direction /= np.linalg.norm(direction)

    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        return Quaternion(0, 0, 0, 1)
    if np.allclose(direction, -z_axis):
        return Quaternion(1, 0, 0, 0)

    rotation_matrix = np.eye(3)
    rotation_matrix[:, 2] = direction
    rotation_matrix[:, 0] = np.cross(z_axis, direction)
    rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
    rotation_matrix[:, 1] = np.cross(direction, rotation_matrix[:, 0])

    rotation = R.from_matrix(rotation_matrix)
    q = rotation.as_quat()
    return Quaternion(q[0], q[1], q[2], q[3])