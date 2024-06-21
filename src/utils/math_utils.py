import os
import random
import math

import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Quaternion

def angle_between_vectors(v1, v2):
  dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
  mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
  mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
  return math.acos(dot_product / (mag_v1 * mag_v2))

def distance_from_current(pose, current_pose):
  return np.sqrt((pose.position.x - current_pose.position.x)**2 + 
                    (pose.position.y - current_pose.position.y)**2 + 
                    (pose.position.z - current_pose.position.z)**2)

def fibonacci_sphere_top_half(samples, radius=1.0, min_distance=0.0, min_z=None, max_z=None):
  points = []
  phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

  for i in range(samples * 10):  # Generate more points to increase the chance of meeting constraints
    y = 1 - (i / float(samples * 10 - 1))  # y goes from 1 to -1
    if y < 0:  # Skip the bottom half of the sphere
      continue
    radius_y = np.sqrt(1 - y * y)  # Radius at y

    theta = phi * i  # Golden angle increment

    x = np.cos(theta) * radius_y
    z = np.sin(theta) * radius_y

    if min_z is not None and z < min_z:
      continue
    if max_z is not None and z > max_z:
      continue

    point = (x * radius, y * radius, z * radius)
    if np.linalg.norm(point) >= min_distance:
      points.append(point)
      if len(points) >= samples:
        break

  return points[:samples]