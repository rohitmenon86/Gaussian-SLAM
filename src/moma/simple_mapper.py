import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos

class Mapper:
  def __init__(self, mobile_manipulator, camera, table_height):
    """
    Initialize the Mapper with the mobile manipulator and camera configuration.

    :param mobile_manipulator: An instance of MobileManipulator
    :param camera: An instance of Camera
    :param table_height: The height of the table from the ground (meters)
    """
    self.mobile_manipulator = mobile_manipulator
    self.camera = camera
    self.table_height = table_height

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
