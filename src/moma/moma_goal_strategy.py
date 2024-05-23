#!/usr/bin/env python

import rospy
import numpy as np
import actionlib
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion
import tf

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion
import tf

class GoalStrategy:
  def get_goal(self):
    raise NotImplementedError("Each strategy must implement the 'get_goal' method.")

class ExternalGoalStrategy(GoalStrategy):
  def __init__(self):
    self.goal = None

  def set_goal(self, x, y, theta):
    self.goal = (x, y, theta)

  def get_goal(self):
    return self.goal

class BorderGoalStrategy(GoalStrategy):
  def __init__(self, navigator):
    self.navigator = navigator
    self.goals = []
  
  def get_goal(self, recompute = True):
    if self.goals == [] or recompute == True:
      self.goals = self.navigator.find_feasible_goals()

    if self.goals == []:
      rospy.logwarn("No goals left")
      return None
    
    goal =  self.goals.pop(0)
    rospy.loginfo("Trying goal at position: {}".format(goal))
    return goal
    


class BorderGoalNavigator:
  def __init__(self, min_distance=0.3, safe_distance = 0.3):  # min_distance in meters
    self.current_map = None
    self.min_distance = min_distance
    self.safe_distance = safe_distance
    self.tf_listener = tf.TransformListener()
    self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
    self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    self.client.wait_for_server()
    rospy.sleep(1)  # Ensure TF buffer is filled

  def map_callback(self, data):
    self.current_map  = data

  def get_current_position(self):
    try:
      (trans, _) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
      return trans
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      rospy.logwarn("Unable to find robot position in map")
      return None
  
  def find_border_goals(self):
    if not self.current_map:
      rospy.loginfo("No map data available.")
      return None, None

    width = self.current_map.info.width
    height = self.current_map.info.height
    resolution = self.current_map.info.resolution
    origin = self.current_map.info.origin.position

    border_points = []
    direction_thetas = []

    for y in range(1, height-1):
      for x in range(1, width-1):
        idx = y * width + x
        if self.current_map.data[idx] == 0:  # Check if the cell is free
          neighbors = [
            self.current_map.data[idx - 1], self.current_map.data[idx + 1],
            self.current_map.data[idx - width], self.current_map.data[idx + width]
          ]
          if -1 in neighbors:  # If adjacent to unknown
            direction_theta = self.calculate_direction_theta(x, y, neighbors)
            if direction_theta is not None:
              x_world = (x * resolution) + origin.x
              y_world = (y * resolution) + origin.y
              border_points.append((x_world, y_world))
              direction_thetas.append(direction_theta)

    if border_points:
      return border_points[0], direction_thetas[0]
    return None, None
  
  def find_closest_border_goal(self):
    if not self.current_map or not self.get_current_position():
      if not self.current_map:
        rospy.loginfo("Map data not available.")
      if not self.get_current_position():
        rospy.loginfo("Current robot pose not available.")
      if not self.current_map and not self.get_current_position():
        rospy.loginfo("Map data AND current robot pose not available.")
      return None, None

    current_position = self.get_current_position()
    min_distance = float('inf')
    closest_goal = None
    closest_theta = None

    width = self.current_map.info.width
    height = self.current_map.info.height
    resolution = self.current_map.info.resolution
    origin = self.current_map.info.origin.position
  
    for y in range(1, height-1):
      for x in range(1, width-1):
        idx = y * width + x
        if self.current_map.data[idx] == 0:  # Free space
          neighbors = [self.current_map.data[idx - 1], self.current_map.data[idx + 1], self.current_map.data[idx - width], self.current_map.data[idx + width]]
          if -1 in neighbors:  # Adjacent to unknown
            x_world = x * resolution + origin.x
            y_world = y * resolution + origin.y
            distance = np.hypot(x_world - current_position[0], y_world - current_position[1])
            if distance < min_distance and distance > self.min_distance:
              min_distance = distance
              closest_goal = (x_world, y_world)
              direction_vector = self.calculate_direction_vector(x, y, neighbors)
              closest_theta = np.arctan2(direction_vector[1], direction_vector[0])
    print("Closest goal and theta: ", closest_goal, closest_theta)
    return closest_goal, closest_theta
  
  def find_feasible_goals(self):
    current_position = self.get_current_position()
    if not self.current_map or not current_position:
      rospy.loginfo("Map data or robot position not available.")
      return []

    goals = []
    width = self.current_map.info.width
    height = self.current_map.info.height
    resolution = self.current_map.info.resolution
    origin = self.current_map.info.origin.position

    grid_x = int((current_position[0] - origin.x) / resolution)
    grid_y = int((current_position[1] - origin.y) / resolution)
    min_distance = float('inf')

    current_position = self.get_current_position()
    if not current_position:
      return []

    for y in range(max(0, grid_y - 50), min(height, grid_y + 50)):
      for x in range(max(0, grid_x - 50), min(width, grid_x + 50)):
        idx = y * width + x
        if self.current_map.data[idx] == 0:  # Free space
          neighbors = [self.current_map.data[n_idx] for n_idx in [idx - 1, idx + 1, idx - width, idx + width] if 0 <= n_idx < width * height]
          if -1 in neighbors:  # Adjacent to unknown
            safe_x = x - int(self.safe_distance / resolution * np.sign(x - grid_x))
            safe_y = y - int(self.safe_distance / resolution * np.sign(y - grid_y))
            safe_idx = safe_y * width + safe_x
            if 0 <= safe_x < width and 0 <= safe_y < height and self.current_map.data[safe_idx] == 0:
              x_world = safe_x * resolution + origin.x
              y_world = safe_y * resolution + origin.y
              distance = np.hypot(x_world - current_position[0], y_world - current_position[1])
              theta = np.arctan2(safe_y - grid_y, safe_x - grid_x)
              goals.append((x_world, y_world, theta, distance))


    # Sort goals by distance to current position
    goals.sort(key=lambda goal: goal[3])
    return [(x, y, theta) for x, y, theta, _ in goals]

  
  def calculate_direction_theta(self, x, y, neighbors):
    vector = np.array([0, 0])
    if neighbors[0] == -1:  # Left is unknown
      vector[0] -= 1
    if neighbors[1] == -1:  # Right is unknown
      vector[0] += 1
    if neighbors[2] == -1:  # Up is unknown
      vector[1] -= 1
    if neighbors[3] == -1:  # Down is unknown
      vector[1] += 1
    
    if np.linalg.norm(vector) == 0:
      return None
    return np.arctan2(vector[1], vector[0])

  def calculate_direction_vector(self, x, y, neighbors):
    vector = [0, 0]
    if neighbors[0] == -1: vector[0] -= 1  # Left
    if neighbors[1] == -1: vector[0] += 1  # Right
    if neighbors[2] == -1: vector[1] -= 1  # Top
    if neighbors[3] == -1: vector[1] += 1  # Bottom
    return vector if np.linalg.norm(vector) != 0 else None

