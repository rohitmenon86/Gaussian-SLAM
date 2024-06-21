#!/usr/bin/env python

import sys
import os
import rospy
import moveit_commander
import actionlib
import yaml
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, PoseArray, Point
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
import random
from trac_ik_python.trac_ik import IK
from utils import conversions, math_utils
import tf
import pyassimp
import xml.etree.ElementTree as ET
import trimesh
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelStates


def load_mesh(mesh_path):
  print("Mesh Path: ", mesh_path)
  scene = pyassimp.load(mesh_path)
  print(scene)
  try:
    if not scene.meshes:
      rospy.logerr(f"Mesh file {mesh_path} could not be loaded")
      return None
      mesh = scene.meshes[0]
      pyassimp.release(scene)
      return mesh
  except:
    rospy.logerr(f"pyassimp scene {mesh_path} has no meshes")
    return None

def get_mesh_path_from_sdf(sdf_path):
  tree = ET.parse(sdf_path)
  root = tree.getroot()

  for visual in root.findall('.//visual'):
    for geometry in visual.findall('geometry'):
      for mesh in geometry.findall('mesh'):
        uri = mesh.find('uri').text
        if uri:
          return uri.replace('file://', '')
  return None

def find_mesh_in_sdf_directory(sdf_path):
  sdf_directory = os.path.dirname(sdf_path)
  meshes_directory = os.path.join(sdf_directory, "meshes")

  if os.path.isdir(meshes_directory):
    for file in os.listdir(meshes_directory):
      if file.endswith(".stl") or file.endswith(".dae"):
        return os.path.join(meshes_directory, file)
  return None

def mesh_to_msg(mesh):
  """Converts a trimesh object to a ROS Mesh message."""
  triangles = []
  for face in mesh.faces:
    triangle = MeshTriangle()
    triangle.vertex_indices = [face[0], face[1], face[2]]
    triangles.append(triangle)
  
  vertices = []
  for vertex in mesh.vertices:
    point = Point()
    point.x, point.y, point.z = vertex
    vertices.append(point)

  mesh_msg = Mesh()
  mesh_msg.triangles = triangles
  mesh_msg.vertices = vertices
  return mesh_msg

def get_bounding_box_and_origin(mesh_path, scale=(1.0, 1.0, 1.0)):
  mesh = trimesh.load(mesh_path)
  
  if isinstance(mesh, trimesh.Scene):
    # If the loaded mesh is a scene, merge all geometries into a single mesh
    mesh = trimesh.util.concatenate(mesh.dump())

  mesh.apply_scale(scale)
  bounding_box = mesh.bounding_box_oriented
  bbox_extents = bounding_box.extents
  bbox_centroid = bounding_box.centroid
  mesh_origin = mesh.centroid

  return bbox_extents, bbox_centroid, mesh_origin

def add_mesh_to_scene(scene, name, pose, mesh_path, scale):
  mesh = trimesh.load(mesh_path)
  
  if isinstance(mesh, trimesh.Scene):
    # If the loaded mesh is a scene, merge all geometries into a single mesh
    mesh = trimesh.util.concatenate(mesh.dump())

  mesh.apply_scale(scale)
  mesh_msg = mesh_to_msg(mesh)

  collision_object = CollisionObject()
  collision_object.id = name
  collision_object.header.frame_id = pose.header.frame_id
  collision_object.meshes = [mesh_msg]
  collision_object.mesh_poses = [pose.pose]
  collision_object.operation = CollisionObject.ADD

  # Add the collision object to the planning scene
  scene.add_object(collision_object)

  rospy.sleep(1)  # Give some time for the object to be added to the planning scene

def get_sdf_path_from_config(config, model_name):
  for table in config['tables']:
    if table['name'] == model_name:
      return table['path']
    for obj in table['objects']:
      if obj['name'] == model_name:
        return obj['path']
  return None

def add_objects_to_scene(scene):
    # Get the list of spawned objects from Gazebo
  model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates)
  
  for i, model_name in enumerate(model_states.name):
    obj_pose = PoseStamped()
    obj_pose.header.frame_id = "world"
    obj_pose.pose = model_states.pose[i]

    # Assuming the sdf_path for each model is stored in a known location
    sdf_path = os.path.join("/home/rohit/.gazebo/models/", model_name, "model.sdf")
    mesh_path = find_mesh_in_sdf_directory(sdf_path)

    if mesh_path:
      scale = (1.0, 1.0, 1.0)  # You may need to get the actual scale if available
      bbox_extents, bbox_centroid, mesh_origin = get_bounding_box_and_origin(mesh_path, scale)
      obj_pose.pose.position.x += bbox_centroid[0]
      obj_pose.pose.position.y += bbox_centroid[1]
      obj_pose.pose.position.z += bbox_centroid[2]
      scene.add_box(model_name, obj_pose, size=bbox_extents)
      rospy.loginfo(f"Added bounding box for {model_name}: extents={bbox_extents}, centroid={bbox_centroid}, origin={mesh_origin}")
    else:
      rospy.logwarn(f"Could not find mesh in {sdf_path}")


def add_objects_to_scene_yaml_topic(scene, config):
  model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates)
  for i, model_name in enumerate(model_states.name):
    obj_pose = PoseStamped()
    obj_pose.header.frame_id = "world"
    obj_pose.pose = model_states.pose[i]
    
    sdf_path = get_sdf_path_from_config(config, model_name)
    if sdf_path is None:
      continue

    mesh_path = find_mesh_in_sdf_directory(sdf_path)
    if mesh_path:
      scale = (1.0, 1.0, 1.0)  # You may need to get the actual scale if available
      bbox_extents, bbox_centroid, mesh_origin = get_bounding_box_and_origin(mesh_path, scale)
      #obj_pose.pose.position.x += bbox_centroid[0]
      #obj_pose.pose.position.y += bbox_centroid[1]
      obj_pose.pose.position.z -= 0.1
      add_mesh_to_scene(scene, model_name, obj_pose, mesh_path, scale)
      rospy.loginfo(f"Added mesh for {model_name}: extents={bbox_extents}, centroid={bbox_centroid}, origin={mesh_origin}")
    else:
      rospy.logwarn(f"Could not find mesh in {sdf_path}")

    for table in config['tables']:
      table_pose = PoseStamped()
      table_pose.header.frame_id = "world"
      table_pose.pose.position.x = table['position']['x']
      table_pose.pose.position.y = table['position']['y']
      table_pose.pose.position.z = 0.2
      table_pose.pose.orientation.x = table['orientation']['x']
      table_pose.pose.orientation.y = table['orientation']['y']
      table_pose.pose.orientation.z = table['orientation']['z']
      table_pose.pose.orientation.w = table['orientation']['w'] 
      table_size = (table['scale']['x'], table['scale']['y'], table['scale']['z'])
      scene.add_box(table['name'], table_pose, size=table_size)

def add_objects_to_scene_yaml(scene, config):
  for table in config['tables']:
    table_pose = PoseStamped()
    table_pose.header.frame_id = "world"
    table_pose.pose.position.x = table['position']['x']
    table_pose.pose.position.y = table['position']['y']
    table_pose.pose.position.z = table['position']['z']
    table_pose.pose.orientation.x = table['orientation']['x']
    table_pose.pose.orientation.y = table['orientation']['y']
    table_pose.pose.orientation.z = table['orientation']['z']
    table_pose.pose.orientation.w = table['orientation']['w']
    table_size = (table['scale']['x'], table['scale']['y'], table['scale']['z'])
    scene.add_box(table['name'], table_pose, size=table_size)

    # Get the absolute z position of the table's top surface
    table_top_z = table_pose.pose.position.z + table_size[2] / 2.0

    # Add objects on this table
    for obj in table['objects']:
      obj_pose = PoseStamped()
      obj_pose.header.frame_id = "world"
      obj_pose.pose.position.x = table_pose.pose.position.x + obj['position']['x']
      obj_pose.pose.position.y = table_pose.pose.position.y + obj['position']['y']
      obj_pose.pose.position.z = table_pose.pose.position.z + obj['position']['z']
      obj_pose.pose.orientation.x = 0.0
      obj_pose.pose.orientation.y = 0.0
      obj_pose.pose.orientation.z = 0.0
      obj_pose.pose.orientation.w = 1.0

      sdf_path = obj['path']
      mesh_path = find_mesh_in_sdf_directory(sdf_path)
      if mesh_path:
        scale = (obj['scale']['x'], obj['scale']['y'], obj['scale']['z'])
        rospy.logwarn(f"Adding mesh in {sdf_path}")
        bbox_extents, bbox_centroid, mesh_origin = get_bounding_box_and_origin(mesh_path, scale)
        obj_pose.pose.position.x += bbox_centroid[0]
        obj_pose.pose.position.y += bbox_centroid[1]
        obj_pose.pose.position.z += bbox_centroid[2]
        scene.add_box(obj['name'], obj_pose, size=bbox_extents)
        rospy.loginfo(f"Added bounding box for {obj['name']}: extents={bbox_extents}, centroid={bbox_centroid}, origin={mesh_origin}")
        #add_mesh_to_scene(scene, obj['name'], obj_pose, mesh_path, scale)
      else:
        rospy.logerr(f"Could not find mesh in {sdf_path}")
      #mesh = load_mesh(mesh_path)
      #if mesh:
