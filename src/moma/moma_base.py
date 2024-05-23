#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import actionlib
import yaml
import tf
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
from src.moma.moma_goal_strategy import GoalStrategy, ExternalGoalStrategy, BorderGoalStrategy, BorderGoalNavigator

class MoMaBase:
  def __init__(self, use_border_goals=False):
    self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    self.client.wait_for_server()
    self.goal_strategy = self.select_strategy(use_border_goals)

  def select_strategy(self, use_border_goals):
    if use_border_goals:
      navigator = BorderGoalNavigator()  # Ensure this is defined or imported
      return BorderGoalStrategy(navigator)
    else:
      return ExternalGoalStrategy()

  def set_external_goal(self, x, y, theta):
        if isinstance(self.goal_strategy, ExternalGoalStrategy):
            self.goal_strategy.set_goal(x, y, theta)

  def move_to_goal(self):
    goal = self.goal_strategy.get_goal()
    if goal:
      x, y, theta = goal
      self._send_goal(x, y, theta)
    else:
      rospy.loginfo("No valid goal received.")

  def _send_goal(self, x, y, theta):
    quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position = Point(x, y, 0)
    goal.target_pose.pose.orientation = Quaternion(*quaternion)
    self.client.send_goal(goal)
    self.client.wait_for_result()

  def move_to_position(self, x, y, theta):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "odom"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x

    goal.target_pose.pose.orientation.z = sin(theta / 2)
    goal.target_pose.pose.orientation.w = cos(theta / 2)

    self.client.send_goal(goal)
    self.client.wait_for_result()
  def get_current_pose(self):
    # Fetch the current pose of the robot from the tf listener
    try:
      (trans, rot) = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
      return trans, rot
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      rospy.logwarn("Failed to get current robot pose")
      return None
    
  def attempt_goal(self, x, y, theta):
    """Attempt to move to the goal and return True if successful."""
    quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position = Point(x, y, 0)
    goal.target_pose.pose.orientation = Quaternion(*quaternion)
    self.client.send_goal(goal)
    self.client.wait_for_result()

    return self.client.get_state() == actionlib.GoalStatus.SUCCEEDED
  
  def autonomous_exploration(self):
    recompute = False
    while not rospy.is_shutdown():
      goal = self.goal_strategy.get_goal(recompute)
      if goal:
        x, y, theta = goal
        if not self.attempt_goal(x, y, theta):
          rospy.logwarn("Failed to reach goal: ({}, {}, {}). Trying next goal...".format(x, y, theta))
          recompute = False
        else:
          rospy.loginfo("Goal reached successfully: ({}, {}, {})".format(x, y, theta))
          recompute = True
      else:
        rospy.loginfo("No more reachable goals or exploration strategy exhausted.")
        break
