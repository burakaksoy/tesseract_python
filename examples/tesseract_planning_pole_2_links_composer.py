import os
import re
import traceback
import numpy as np
import time
import sys
from pathlib import Path

from tesseract_robotics.tesseract_common import FilesystemPath, \
                                                Isometry3d, \
                                                Translation3d, \
                                                Quaterniond, \
                                                ManipulatorInfo, \
                                                GeneralResourceLocator, \
                                                CollisionMarginData, \
                                                AnyPoly, \
                                                AnyPoly_wrap_double, \
                                                ResourceLocator, \
                                                SimpleLocatedResource, \
                                                TransformMap, \
                                                CONSOLE_BRIDGE_LOG_DEBUG, \
                                                setLogLevel, \
                                                Timer, \
                                                AngleAxisd

from tesseract_robotics.tesseract_environment import Environment, \
                                                     AddLinkCommand, \
                                                     Commands 

from tesseract_robotics.tesseract_scene_graph import Joint, \
                                                     Link, \
                                                     Visual, \
                                                     Collision, \
                                                     JointType_FIXED, \
                                                     Material

from tesseract_robotics.tesseract_geometry import Sphere, \
                                                    Box, \
                                                    Cylinder, \
                                                    ConvexMesh, \
                                                    Mesh, \
                                                    Plane, \
                                                    MeshMaterial

from tesseract_robotics.tesseract_command_language import CartesianWaypoint, \
                                                          CartesianWaypointPoly, \
                                                          CartesianWaypointPoly_wrap_CartesianWaypoint, \
                                                          WaypointPoly, \
                                                          JointWaypoint, \
                                                          JointWaypointPoly, \
                                                          StateWaypointPoly_wrap_StateWaypoint, \
                                                          JointWaypointPoly_wrap_JointWaypoint, \
                                                          InstructionPoly, \
                                                          MoveInstruction, \
                                                          MoveInstructionPoly, \
                                                          MoveInstructionType_FREESPACE, \
                                                          MoveInstructionType_LINEAR, \
                                                          MoveInstructionType_CIRCULAR, \
                                                          MoveInstructionPoly_wrap_MoveInstruction, \
                                                          InstructionPoly_as_MoveInstructionPoly, \
                                                          ProfileDictionary, \
                                                          WaypointPoly_as_StateWaypointPoly, \
                                                          StateWaypoint, \
                                                          StateWaypointPoly, \
                                                          CompositeInstruction, \
                                                          AnyPoly_as_CompositeInstruction, \
                                                          AnyPoly_wrap_CompositeInstruction, \
                                                          CompositeInstructionOrder_ORDERED, \
                                                          DEFAULT_PROFILE_KEY, \
                                                          toJointTrajectory

from tesseract_robotics_viewer import TesseractViewer
                                      

from tesseract_robotics.tesseract_task_composer import  TaskComposerPluginFactory, \
                                                        PlanningTaskComposerProblem, \
                                                        PlanningTaskComposerProblemUPtr, \
                                                        PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr, \
                                                        TaskComposerDataStorage, \
                                                        TaskComposerContext, \
                                                        TaskComposerFuture, \
                                                        TaskComposerFutureUPtr, \
                                                        MinLengthProfile, \
                                                        ProfileDictionary_addProfile_MinLengthProfile
                                                        
                                                        
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram

from tesseract_robotics.tesseract_motion_planners_ompl import OMPLDefaultPlanProfile, \
                                                              RRTConnectConfigurator, \
                                                              OMPLProblemGeneratorFn, \
                                                              OMPLMotionPlanner, \
                                                              ProfileDictionary_addProfile_OMPLPlanProfile

from tesseract_robotics.tesseract_motion_planners_trajopt import TrajOptDefaultPlanProfile,\
                                                                 TrajOptPlanProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptPlanProfile, \
                                                                 TrajOptDefaultCompositeProfile, \
                                                                 TrajOptCompositeProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptCompositeProfile, \
                                                                 TrajOptDefaultSolverProfile, \
                                                                 TrajOptSolverProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptSolverProfile, \
                                                                 TrajOptProblemGeneratorFn, \
                                                                 TrajOptMotionPlanner, \
                                                                 CollisionEvaluatorType_SINGLE_TIMESTEP, \
                                                                 CollisionEvaluatorType_DISCRETE_CONTINUOUS, \
                                                                 CollisionEvaluatorType_CAST_CONTINUOUS, \
                                                                 ModelType, \
                                                                 BasicTrustRegionSQPParameters
                                                                 
                                                                 
from tesseract_robotics.tesseract_collision import ContactTestType_ALL, \
                                                   ContactTestType_FIRST, \
                                                   ContactTestType_CLOSEST
                                                   


from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, \
                                                               InstructionsTrajectory
                                                        

import tf.transformations as tf_trans

# This example demonstrates using the Tesseract Planning Task Composer to create a simple robot motion plan from
# an input request program. The composer is a high level interface for creating motion plans. It is designed to
# be used by users who do not want to deal with the details of the Tesseract Planning Library. The composer automatically
# runs a sequence of planning steps to generate an output plan with minimal configuration. "Profiles" are used to
# configure the planning steps. Profiles are a dictionary of key value pairs that are used to configure the planning
# steps. The various planners have default configurations that should work for most use cases. There are numerous
# configurations available for the task composer that execute different sequences of planning steps. This example 
# demonstrates using the "freespace" planner, which is for moving the robot to a desired pose in free space while
# avoiding collisions. The freespace planner first uses OMPL to find a collision free path, and then uses TrajOpt
# to refine the path. Finally, the TimeOptimalTrajectoryGeneration time parametrization algorithm is used to generate
# timestamps for the trajectory.

# The task composer requires a configuration YAML file to be passed in. A default configuration file is provided
# in the Tesseract Planning Library. This configuration file can be copied and modified to suit the user's needs.

# The environmental variable TESSERACT_TASK_COMPOSER_CONFIG_FILE must be set to the location of the configuration file.
# This can be done by running:
#
# git clone https://github.com/tesseract-robotics/tesseract-planning.git
# export TESSERACT_TASK_COMPOSER_CONFIG_FILE="$(pwd)/tesseract_planning/tesseract_task_composer/config/task_composer_plugins_no_trajopt_ifopt.yaml"

# An environment is initialized using URDF and SRDF files. These files need to be configured for the scene, and
# to use the correct collision and kinematics plugins. See the collision and kinematics examples for more details on
# how to do this.

# This example uses the GeneralResourceLocator to find resources on the file system. The GeneralResourceLocator
# uses the TESSERACT_RESOURCE_PATH environmental variable.
#
# TESSERACT_RESOURCE_PATH must be set to the directory containing the `tesseract_support` package. This can be done
# by running:
#
# git clone https://github.com/tesseract-robotics/tesseract.git
# export TESSERACT_RESOURCE_PATH="$(pwd)/tesseract/"

def add_environment_obstacles(env, viewer, obstacle_ceiling_active=True):
    cmds = Commands()
    # --------------------------------------------------------------------------------------------
    # ADD ENVIRONMENT OBJECTS HERE

    # # Add a sphere using Environment commands
    # sphere_link_visual = Visual()
    # sphere_link_visual.geometry = Sphere(0.1)

    # sphere_link_collision = Collision()
    # sphere_link_collision.geometry = Sphere(0.1)

    # sphere_link = Link("sphere_link")
    # sphere_link.visual.push_back(sphere_link_visual)
    # sphere_link.collision.push_back(sphere_link_collision)

    # sphere_joint = Joint("sphere_joint")
    # sphere_joint.parent_link_name = "base_link"
    # sphere_joint.child_link_name = sphere_link.getName()
    # sphere_joint.type = JointType_FIXED

    # sphere_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, 1.0)
    # sphere_joint.parent_to_joint_origin_transform = sphere_link_joint_transform

    # add_sphere_command = AddLinkCommand(sphere_link, sphere_joint)
    # env.applyCommand(add_sphere_command)

    viewer.update_environment(env, [0,0,0])

    # OBSTACLE BOX
    # Add a box using Environment commands
    box_link_visual = Visual()
    box_link_visual.geometry = Box(1.5, 1, 1)
    box_link_visual.material = Material()
    box_link_visual.material.color = np.array([0.4, 0.4, 0.4, 0.5])

    box_link_collision = Collision()
    box_link_collision.geometry = Box(1.5, 1, 1)

    box_link = Link("box_link")
    box_link.visual.push_back(box_link_visual)
    box_link.collision.push_back(box_link_collision)

    box_joint = Joint("box_joint")
    box_joint.parent_link_name = "base_link"
    box_joint.child_link_name = box_link.getName()
    box_joint.type = JointType_FIXED

    box_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 1.0, 0.5)
    box_joint.parent_to_joint_origin_transform = box_link_joint_transform

    add_box_command = AddLinkCommand(box_link, box_joint)
    cmds.push_back(add_box_command)
    # env.applyCommand(add_box_command)

    viewer.update_environment(env, [0,0,0])

    # GROUND PLANE (Assume square plane)
    plane_w = 6.0
    plane_h = 0.1

    # Add a box using Environment commands as a ground plane
    plane_link_visual = Visual()
    plane_link_visual.geometry = Box(plane_w,plane_w,plane_h)
    plane_link_visual.material = Material()
    plane_link_visual.material.color = np.array([0.5, 0.5, 0.5, 1.0])

    plane_link_collision = Collision()
    plane_link_collision.geometry = Box(plane_w,plane_w,plane_h)

    plane_link = Link("plane_link")
    plane_link.visual.push_back(plane_link_visual)
    plane_link.collision.push_back(plane_link_collision)

    plane_joint = Joint("plane_joint")
    plane_joint.parent_link_name = "base_link"
    plane_joint.child_link_name = plane_link.getName()
    plane_joint.type = JointType_FIXED

    plane_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, -plane_h/2.0)
    plane_joint.parent_to_joint_origin_transform = plane_link_joint_transform

    add_plane_command = AddLinkCommand(plane_link, plane_joint)
    cmds.push_back(add_plane_command)
    # env.applyCommand(add_plane_command)

    viewer.update_environment(env, [0,0,0])

    # OBSTACLE CEILING
    # obstacle_ceiling_active = True
    ceil_angle = np.pi/6.0
    ceil_min_h = 0.0 # 0.2
    ceil_l = (plane_w/2.0)/np.cos(ceil_angle)

    if obstacle_ceiling_active:
        # Add a box using Environment commands as a ceiling
        ceiling_link_visual = Visual()
        ceiling_link_visual.geometry = Box(ceil_l, plane_w, plane_h)
        ceiling_link_visual.material = Material()
        ceiling_link_visual.material.color = np.array([0.5, 0.5, 0.5, 1.0])

        ceiling_link_collision = Collision()
        ceiling_link_collision.geometry = Box(ceil_l, plane_w, plane_h)

        ceiling_link = Link("ceiling_link")
        ceiling_link.visual.push_back(ceiling_link_visual)
        ceiling_link.collision.push_back(ceiling_link_collision)

        ceiling_joint = Joint("ceiling_joint")
        ceiling_joint.parent_link_name = "base_link"
        ceiling_joint.child_link_name = ceiling_link.getName()
        ceiling_joint.type = JointType_FIXED

        x = plane_w/4.0
        z = ceil_min_h + ceil_l*np.sin(ceil_angle)/2.0
        ceiling_link_joint_transform = Isometry3d.Identity() * \
                                    Translation3d(x, 0.0, z) * \
                                    AngleAxisd(ceil_angle, np.array([0.0,1.0,0.0]))
        
        ceiling_joint.parent_to_joint_origin_transform = ceiling_link_joint_transform

        add_ceiling_command = AddLinkCommand(ceiling_link, ceiling_joint)
        cmds.push_back(add_ceiling_command)
        # env.applyCommand(add_ceiling_command)

        viewer.update_environment(env, [0,0,0])

        # Add the other half of the ceiling
        ceiling_link2 = Link("ceiling_link2")
        ceiling_link2.visual.push_back(ceiling_link_visual)
        ceiling_link2.collision.push_back(ceiling_link_collision)

        ceiling_joint2 = Joint("ceiling_joint2")
        ceiling_joint2.parent_link_name = "base_link"
        ceiling_joint2.child_link_name = ceiling_link2.getName()
        ceiling_joint2.type = JointType_FIXED

        ceiling_link_joint_transform2 = Isometry3d.Identity() * \
                                        Translation3d(-x, 0.0, z) * \
                                        AngleAxisd(-ceil_angle, np.array([0.0,1.0,0.0]))
        
        ceiling_joint2.parent_to_joint_origin_transform = ceiling_link_joint_transform2

        add_ceiling_command2 = AddLinkCommand(ceiling_link2, ceiling_joint2)
        cmds.push_back(add_ceiling_command2)
        # env.applyCommand(add_ceiling_command2)

        viewer.update_environment(env, [0,0,0])

    env.applyCommands(cmds)
    viewer.update_environment(env, [0,0,0])

    # --------------------------------------------------------------------------------------------

setLogLevel(CONSOLE_BRIDGE_LOG_DEBUG)

task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]

# Initialize the resource locator and environment
locator = GeneralResourceLocator()
urdf_package_url = "package://tesseract_support/urdf/dlo_description/urdf/pole_2_links.urdf"
srdf_package_url = "package://tesseract_support/urdf/dlo_description/urdf/pole_2_links.srdf"
urdf_fname = FilesystemPath(locator.locateResource(urdf_package_url).getFilePath())
srdf_fname = FilesystemPath(locator.locateResource(srdf_package_url).getFilePath())

env = Environment()

# locator_fn must be kept alive by maintaining a reference
assert env.init(urdf_fname, srdf_fname, locator)

# Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
# match the SRDF, although the exact tcp_frame can differ if a tool is used.
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "tool1" # "cylindrical_1_end1_link" #"cylindrical_1_end1_link" # "cylindrical_2_end1_link" # "cylindrical_left_1_end1_link" # "tool0"
manip_info.manipulator = "manipulator"
manip_info.working_frame = "base_link"

# Create a viewer and set the environment so the results can be displayed later
viewer = TesseractViewer()
# Show the world coordinate frame
viewer.add_axes_marker(position=[0,0,0], quaternion=[1,0,0,0], size=1.0, parent_link="base_link", name="world_frame")


viewer.update_environment(env, [0,0,0])

# Set the initial state of the robot
# joint_names = ["cartesian_x_joint",
#                "cartesian_y_joint",
#                "cartesian_z_joint",
#                "spherical_yaw_joint",
#                "spherical_pitch_joint",
#                "spherical_roll_joint",
#                "cylindrical_1_to_cylindrical_2"]

# Alternatively, get the joint names from the environment:
joint_group = env.getJointGroup("manipulator")
joint_names = list(joint_group.getJointNames())

# Another way to get the joint names:
# joint_names = list(env.getGroupJointNames("manipulator"))

print("joint_names: ", joint_names)
print("")

# some other ENV GET methods:
# """
all_joint_names = list(env.getJointNames())
print("all_joint_names: ", all_joint_names)
print("")

all_link_names = list(env.getLinkNames())
print("all_link_names: ", all_link_names)
print("")

active_joint_names = list(env.getActiveJointNames())
print("active_joint_names: ", active_joint_names)
print("")

active_link_names = list(env.getActiveLinkNames())
print("active_link_names: ", active_link_names)
print("")

current_joint_values = env.getCurrentJointValues()
print("current_joint_values: ", current_joint_values)
print("")

group_names = list(env.getGroupNames())
print("group_names: ", group_names)
print("")

root_link_name = env.getRootLinkName()
print("root_link_name: ", root_link_name)
print("")

static_link_names = list(env.getStaticLinkNames())
print("static_link_names: ", static_link_names)
print("")
# """


# Set the initial state of the robot
pos_x = 0 # -3.3528/2.0 # 0

init_x = pos_x # 0
init_y = 2.0
init_z = 0.5

init_yaw = np.pi/2
init_pitch = 0
init_roll = np.pi/2

init_q = tf_trans.quaternion_from_euler(*[init_roll, init_pitch, init_yaw], 'sxyz') # in xyzw format!!
init_q = [init_q[3], init_q[0], init_q[1], init_q[2]] # convert to wxyz format

# Add the initial pose to the viewer
viewer.add_axes_marker(position=[init_x,init_y,init_z], quaternion=init_q, size=0.5, parent_link="base_link", name="init_frame")

# Set the goal state of the robot
goal_x = pos_x # 0
goal_y = -1.0
goal_z = 0.16

goal_yaw = np.pi/2
goal_pitch = 0
goal_roll = np.pi/2

goal_q = tf_trans.quaternion_from_euler(*[goal_roll, goal_pitch, goal_yaw], 'sxyz') # in xyzw format!!
goal_q = [goal_q[3], goal_q[0], goal_q[1], goal_q[2]] # convert to wxyz format

# Add the goal pose to the viewer
viewer.add_axes_marker(position=[goal_x,goal_y,goal_z], quaternion=goal_q, size=0.5, parent_link="base_link", name="goal_frame")

# Define some intermediate points
i0_x = pos_x # 0
i0_y = 1.5
i0_z = 1.5
i0_yaw = np.pi/2.0 
i0_pitch = 0.0
i0_roll = np.pi/2.0 + np.pi/6.0
# i0_roll = np.pi/2.0

i1_x = pos_x # 0
i1_y = 1.0
i1_z = 1.1 # 0.9
i1_yaw = np.pi/2.0 
i1_pitch = 0.0
# i1_roll = np.pi/2.0 + np.pi/6.0 
i1_roll = np.pi/2.0

i2_x = pos_x # 0
i2_y = 0.5
i2_z = 1.5
i2_yaw = np.pi/2.0 
i2_pitch = 0.0
i2_roll = np.pi/2.0 + np.pi/6.0 
# i2_roll = np.pi/2.0


# initial_joint_positions = np.array([init_x, init_y, init_z, init_yaw, init_pitch, init_roll])
initial_joint_positions = np.array([init_x, init_y, init_z, init_yaw, init_pitch, init_roll, 0.0])

viewer.update_joint_positions(joint_names, initial_joint_positions)
# Start the viewer
viewer.start_serve_background()

# add_environment_obstacles(env, viewer, obstacle_ceiling_active=False)
add_environment_obstacles(env, viewer, obstacle_ceiling_active=True)

# # ------------
# # Set the initial state of the robot
# env.setState(joint_names, initial_joint_positions)

# # Get the state solver. This must be called again after environment is updated
# solver = env.getStateSolver()

# # Get the discrete contact manager. This must be called again after the environment is updated
# manager = env.getDiscreteContactManager()
# manager.setActiveCollisionObjects(env.getActiveLinkNames())

# # Set the collision margin for check. Objects with closer than the specified margin will be returned
# margin_data = CollisionMarginData(0.03) # 3cm margin
# manager.setCollisionMarginData(margin_data)

# # Set the transform of the active collision objects from SceneState
# solver.setState(joint_names, initial_joint_positions)
# scene_state = solver.getState()
# manager.setCollisionObjectsTransform(scene_state.link_transforms)
# # ------------




# --------------------------------------------------------------------------------------------
# """

# Create the input command program waypoints
# wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(init_x,init_y,init_z) * Quaterniond(*init_q))
wp1 = StateWaypoint(joint_names, initial_joint_positions)

# wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(goal_x,goal_y,goal_z) * Quaterniond(*goal_q))
# wp2 = StateWaypoint(joint_names, np.array([goal_x, goal_y, goal_z, goal_yaw, goal_pitch, goal_roll]))
wp2 = StateWaypoint(joint_names, np.array([goal_x, goal_y, goal_z, goal_yaw, goal_pitch, goal_roll, 0.0]))

# wp_i0 = StateWaypoint(joint_names, np.array([i0_x, i0_y, i0_z, i0_yaw, i0_pitch, i0_roll]))
# wp_i0 = StateWaypoint(joint_names, np.array([i0_x, i0_y, i0_z, i0_yaw, i0_pitch, i0_roll, 0.0]))
wp_i0 = StateWaypoint(joint_names, np.array([i0_x, i0_y, i0_z, i0_yaw, i0_pitch, i0_roll, -np.pi/3]))

# wp_i1 = StateWaypoint(joint_names, np.array([i1_x, i1_y, i1_z, i1_yaw, i1_pitch, i1_roll]))
wp_i1 = StateWaypoint(joint_names, np.array([i1_x, i1_y, i1_z, i1_yaw, i1_pitch, i1_roll, 0.0]))
# wp_i1 = StateWaypoint(joint_names, np.array([i1_x, i1_y, i1_z, i1_yaw, i1_pitch, i1_roll, -np.pi/3]))

# wp_i2 = StateWaypoint(joint_names, np.array([i2_x, i2_y, i2_z, i2_yaw, i2_pitch, i2_roll]))
# wp_i2 = StateWaypoint(joint_names, np.array([i2_x, i2_y, i2_z, i2_yaw, i2_pitch, i2_roll, 0.0]))
wp_i2 = StateWaypoint(joint_names, np.array([i2_x, i2_y, i2_z, i2_yaw, i2_pitch, i2_roll, -np.pi/3]))

# Create the input command program instructions. Note the use of explicit construction of the CartesianWaypointPoly
# using the *_wrap_CartesianWaypoint functions. This is required because the Python bindings do not support implicit
# conversion from the CartesianWaypoint to the CartesianWaypointPoly.

# start_instruction = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
start_instruction = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
# start_instruction = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp1), MoveInstructionType_FREESPACE, "freespace_profile")

plan_i0 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i0), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_i0 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i0), MoveInstructionType_FREESPACE, "freespace_profile")

plan_i1 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i1), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_i1 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i1), MoveInstructionType_FREESPACE, "freespace_profile")

plan_i2 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i2), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_i2 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp_i2), MoveInstructionType_FREESPACE, "freespace_profile")

# plan_f1 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")
plan_f1 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_f1 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp2), MoveInstructionType_FREESPACE, "freespace_profile")

# Create the input command program. Note the use of *_wrap_MoveInstruction functions. This is required because the
# Python bindings do not support implicit conversion from the MoveInstruction to the MoveInstructionPoly.
program = CompositeInstruction("DEFAULT")
# program = CompositeInstruction("freespace_profile")
program.setManipulatorInfo(manip_info)

## Add the instructions to the program
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))

# program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_i0))

# program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_i1))

# program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_i2))

program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))

# Print diagnosics
program._print("Program: ")

## Create an AnyPoly containing the program. This explicit step is required because the Python bindings do not
# support implicit conversion from the CompositeInstruction to the AnyPoly.
program_anypoly = AnyPoly_wrap_CompositeInstruction(program)

# Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
# in the instructions.
profiles = ProfileDictionary()

# --------------------------------------------------------------------------------------------
MINLENGTH_DEFAULT_NAMESPACE = "MinLengthTask"

# Set the number of steps to use for the trajectory for the MinLengthProfile
min_length_profile = MinLengthProfile(60)

ProfileDictionary_addProfile_MinLengthProfile(profiles, MINLENGTH_DEFAULT_NAMESPACE, "DEFAULT", min_length_profile)

# --------------------------------------------------------------------------------------------

# ----------------------------------------------------------
# """
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"

####################
## Plan Profile BEGIN

trajopt_plan_profile = TrajOptDefaultPlanProfile()

# trajopt_plan_profile.cartesian_coeff = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
# trajopt_plan_profile.joint_coeff = np.array([10, 10, 1, 100, 100, 100, 1], dtype=np.float64)
# trajopt_plan_profile.joint_coeff = np.array([1000, 1000, 0, 100, 100, 100, 0], dtype=np.float64)
# trajopt_plan_profile.joint_coeff = np.array([0,0,0,0,0,0,0], dtype=np.float64)

# trajopt_plan_profile.constraint_error_functions # ???

# trajopt_plan_profile.term_type # ???

# Arguments: (profile_dictionary, ns, profile_name, profile)
# ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_plan_profile)
ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
# ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "freespace_profile", trajopt_plan_profile)

## Plan Profile END
####################


####################
## Composite Profile BEGIN 

trajopt_composite_profile = TrajOptDefaultCompositeProfile()

trajopt_composite_profile.collision_cost_config.enabled = True # If true, a collision cost term will be added to the problem. Default: true*/
trajopt_composite_profile.collision_cost_config.use_weighted_sum = False # Use the weighted sum for each link pair. This reduces the number equations added to the problem. If set to true, it is recommended to start with the coeff set to one Default: false*/
trajopt_composite_profile.collision_cost_config.safety_margin = 0.0150 # 2.5cm #  Max distance in which collision costs will be evaluated. Default: 0.025*/
trajopt_composite_profile.collision_cost_config.safety_margin_buffer = 0.00 # Distance beyond buffer_margin in which collision optimization will be evaluated. This is set to 0 by default (effectively disabled) for collision costs.
trajopt_composite_profile.collision_cost_config.type = CollisionEvaluatorType_DISCRETE_CONTINUOUS # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
trajopt_composite_profile.collision_cost_config.coeff = 0.1 # The collision coeff/weight. Default: 20*/

trajopt_composite_profile.collision_constraint_config.enabled = True # If true, a collision cost term will be added to the problem. Default: true
trajopt_composite_profile.collision_constraint_config.use_weighted_sum = False # Use the weighted sum for each link pair. This reduces the number equations added to the problem. If set to true, it is recommended to start with the coeff set to one. Default: false
trajopt_composite_profile.collision_constraint_config.safety_margin = 0.016 # Max distance in which collision costs will be evaluated. Default: 0.01
trajopt_composite_profile.collision_constraint_config.safety_margin_buffer = 0.00 # Distance beyond buffer_margin in which collision optimization will be evaluated. Default: 0.05
trajopt_composite_profile.collision_constraint_config.type = CollisionEvaluatorType_DISCRETE_CONTINUOUS # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
trajopt_composite_profile.collision_constraint_config.coeff = 1 # The collision coeff/weight. Default: 20

# The type of contact test to perform: FIRST, CLOSEST, ALL. Default: ALL
trajopt_composite_profile.contact_test_type = ContactTestType_ALL # ContactTestType_CLOSEST 

trajopt_composite_profile.smooth_velocities = True # If true, a joint velocity cost with a target of 0 will be applied for all timesteps Default: true
# trajopt_composite_profile.velocity_coeff = np.array([10, 10, 1, 100, 100, 100, 1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

trajopt_composite_profile.smooth_accelerations = False # If true, a joint acceleration cost with a target of 0 will be applied for all timesteps Default: false
# trajopt_composite_profile.acceleration_coeff = np.array([1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

trajopt_composite_profile.smooth_jerks = False # If true, a joint jerk cost with a target of 0 will be applied for all timesteps Default: false
# trajopt_composite_profile.jerk_coeff = np.array([1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

trajopt_composite_profile.avoid_singularity = False #  If true, applies a cost to avoid kinematic singularities. Default: false
trajopt_composite_profile.avoid_singularity_coeff = 5.0 # Optimization weight associated with kinematic singularity avoidance. Default: 5.0

trajopt_composite_profile.longest_valid_segment_fraction = 0.01 # Set the resolution at which state validity needs to be verified in order for a motion between two states to be considered valid in post checking of trajectory returned by trajopt. The resolution is equal to longest_valid_segment_fraction * state_space.getMaximumExtent(). Default: 0.01
# Note: The planner takes the conservative of either longest_valid_segment_fraction or longest_valid_segment_length.

trajopt_composite_profile.longest_valid_segment_length = 0.1 # Set the resolution at which state validity needs to be verified in order for a motion between two states to be considered valid. If norm(state1 - state0) > longest_valid_segment_length. Default: 0.1
# Note: This gets converted to longest_valid_segment_fraction. longest_valid_segment_fraction = longest_valid_segment_length / state_space.getMaximumExtent()

# Arguments: (profile_dictionary, ns, profile_name, profile)
# ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_composite_profile)
ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)
# ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "freespace_profile", trajopt_composite_profile)

## Composite Profile END
####################


####################
## Solver Profile BEGIN
trajopt_solver_profile = TrajOptDefaultSolverProfile()

modelType = ModelType()
# modelType.OSQP
trajopt_solver_profile.convex_solver.type = modelType.OSQP # GUROBI # The convex solver to use. Default: OSQP
# trajopt_solver_profile.convex_solver.tolerance = 1e-6 # The tolerance for the solver. Default: 1e-5

# trajopt_solver_profile.convex_solver_config =  ?? # The convex solver config to use, if nullptr the default settings are used

trajopt_solver_profile.opt_info = BasicTrustRegionSQPParameters() # Optimization paramters */sco::BasicTrustRegionSQPParameters
trajopt_solver_profile.opt_info.max_iter = 100000 # The maximum number of iterations
# trajopt_solver_profile.opt_info.min_approx_improve = 1e-3 # The minimum approximate improvement
# trajopt_solver_profile.opt_info.min_trust_box_size = 1e-3 # The minimum trust region size


# Arguments: (profile_dictionary, ns, profile_name, profile)
# ProfileDictionary_addProfile_TrajOptSolverProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_solver_profile)
ProfileDictionary_addProfile_TrajOptSolverProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_solver_profile)
# ProfileDictionary_addProfile_TrajOptSolverProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "freespace_profile", trajopt_solver_profile)

## Solver Profile END
####################
# """
# ----------------------------------------------------------

# Create the task problem and input
task_planning_problem = PlanningTaskComposerProblem(env, profiles)
task_planning_problem.input = program_anypoly

task_planning_problem.profiles

# Create the task composer plugin factory and load the plugins
config_path = FilesystemPath(task_composer_filename)
factory = TaskComposerPluginFactory(config_path)

# Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
task = factory.createTaskComposerNode("FreespacePipeline")
# task = factory.createTaskComposerNode("TrajOptPipeline")
# task = factory.createTaskComposerNode("OMPLPipeline")

# task = factory.createTaskComposerNode("FreespacePipeline2") # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation
# task = factory.createTaskComposerNode("TrajOptPipeline2") # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation

# Get the output keys for the task
output_key = task.getOutputKeys()[0]

# Create an executor to run the task
task_executor = factory.createTaskComposerExecutor("TaskflowExecutor")

# Run the task and wait for completion
future = task_executor.run(task.get(), task_planning_problem)
future.wait()

# Retrieve the output, converting the AnyPoly back to a CompositeInstruction
results = AnyPoly_as_CompositeInstruction(future.context.data_storage.getData(output_key))


# Display the output
# Print out the resulting waypoints
for instr in results:
    assert instr.isMoveInstruction()
    move_instr1 = InstructionPoly_as_MoveInstructionPoly(instr)
    wp1 = move_instr1.getWaypoint()
    assert wp1.isStateWaypoint()
    wp = WaypointPoly_as_StateWaypointPoly(wp1)
    print("-------------------------------------------------------------")
    print(f"Joint Time: {wp.getTime()}")
    print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")
    # print("Joint Names: " + str(list(wp.getNames())))
    # print(f"Joint Velocities: {wp.getVelocity().flatten()}")
    # print(f"Joint Accelerations: {wp.getAcceleration().flatten()}")
    # print(f"Joint Efforts: {wp.getEffort().flatten()}")
    print("-------------------------------------------------------------")


# Update the viewer with the results to animate the trajectory
# Open web browser to http://localhost:8000 to view the results
viewer.update_trajectory(results)
viewer.plot_trajectory(results, manip_info)

# """
# --------------------------------------------------------------------------------------------

input("press enter to exit")