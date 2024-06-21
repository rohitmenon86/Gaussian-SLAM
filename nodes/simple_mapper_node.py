#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from src.moma.simple_mapper import MobileManipulator, Camera, Mapper

def main():
    rospy.init_node('mobile_manipulator_navigation')

    # Initialize components
    mobile_manipulator = MobileManipulator('arm_group_name')
    camera = Camera([525, 525, 319.5, 239.5], 60, 45, (640, 480))
    mapper = Mapper(mobile_manipulator, camera, 0.75)  # Assuming table height is 0.75 meters

    # Set up subscribers and services
    rospy.Subscriber("/table_detected", PoseStamped, table_callback)
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    move_base_client.wait_for_server()

    rospy.spin()

def move_to_position(x, y, theta):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.z = sin(theta / 2)
    goal.target_pose.pose.orientation.w = cos(theta / 2)
    
    move_base_client.send_goal(goal)
    move_base_client.wait_for_result()

def table_callback(msg):
    rospy.loginfo("Table detected at position: %s", msg.pose.position)
    # Assume table detection includes pose where the table is perfectly perceived
    viewpoints = mapper.find_viewpoints()
    for viewpoint in viewpoints:
        if mobile_manipulator.arm.is_pose_reachable(viewpoint):
            rospy.loginfo("Moving to viewpoint: %s", viewpoint)
            move_to_position(viewpoint[0], viewpoint[1], 0)  # Assuming the theta is 0 for simplicity
            # Perform the mapping from this position
            # Assume mapping functionality or a separate service call
        else:
            rospy.logwarn("Viewpoint not reachable: %s", viewpoint)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
