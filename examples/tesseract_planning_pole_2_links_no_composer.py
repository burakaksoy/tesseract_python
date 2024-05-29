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
                                                Timer

from tesseract_robotics.tesseract_environment import Environment, \
                                                     AddLinkCommand

from tesseract_robotics.tesseract_scene_graph import Joint, \
                                                     Link, \
                                                     Visual, \
                                                     Collision, \
                                                     JointType_FIXED

from tesseract_robotics.tesseract_geometry import Sphere, \
                                                    Box, \
                                                    Cylinder, \
                                                    ConvexMesh, \
                                                    Mesh, \
                                                    Plane

from tesseract_robotics.tesseract_command_language import CartesianWaypoint, \
                                                          WaypointPoly, \
                                                          MoveInstruction, \
                                                          MoveInstructionType_FREESPACE, \
                                                          MoveInstructionType_LINEAR, \
                                                          InstructionPoly, \
                                                          CompositeInstruction, \
                                                          MoveInstructionPoly, \
                                                          CartesianWaypointPoly, \
                                                          ProfileDictionary, \
                                                          CartesianWaypointPoly_wrap_CartesianWaypoint, \
                                                          MoveInstructionPoly_wrap_MoveInstruction, \
                                                          InstructionPoly_as_MoveInstructionPoly, \
                                                          WaypointPoly_as_StateWaypointPoly, \
                                                          StateWaypoint, \
                                                          StateWaypointPoly, \
                                                          MoveInstructionPoly, \
                                                          AnyPoly_as_CompositeInstruction, \
                                                          AnyPoly_wrap_CompositeInstruction, \
                                                          CompositeInstructionOrder_ORDERED, \
                                                          DEFAULT_PROFILE_KEY, \
                                                          JointWaypoint, \
                                                          JointWaypointPoly, \
                                                          StateWaypointPoly_wrap_StateWaypoint, \
                                                          JointWaypointPoly_wrap_JointWaypoint, \
                                                          toJointTrajectory

from tesseract_robotics_viewer import TesseractViewer

from tesseract_robotics.tesseract_motion_planners import PlannerRequest, \
                                                         PlannerResponse, \
                                                         toToolpath

from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram

from tesseract_robotics.tesseract_motion_planners_ompl import OMPLDefaultPlanProfile, \
                                                              RRTConnectConfigurator, \
                                                              OMPLProblemGeneratorFn, \
                                                              OMPLMotionPlanner, \
                                                              ProfileDictionary_addProfile_OMPLPlanProfile

from tesseract_robotics.tesseract_motion_planners_trajopt import TrajOptDefaultPlanProfile,\
                                                                 TrajOptDefaultCompositeProfile, \
                                                                 TrajOptProblemGeneratorFn, \
                                                                 TrajOptMotionPlanner, \
                                                                 ProfileDictionary_addProfile_TrajOptPlanProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptCompositeProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptSolverProfile, \
                                                                 TrajOptCompositeProfile, \
                                                                 TrajOptDefaultSolverProfile, \
                                                                 TrajOptPlanProfile, \
                                                                 TrajOptSolverProfile

from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, \
                                                               InstructionsTrajectory

import tf.transformations as tf_trans


# This example demonstrates using the Tesseract Planners without using the Tesseract Composer. In most cases it is
# recommended to use the Tesseract Composer as it provides a more robust and flexible interface. However, there are
# cases where the Tesseract Composer is not available or it is desired to use the Tesseract Planners without the
# Tesseract Composer. This example demonstrates how to do that.

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


OMPL_DEFAULT_NAMESPACE = "OMPLMotionPlannerTask"
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"

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
manip_info.tcp_frame = "cylindrical_2_end1_link" # "cylindrical_left_1_end1_link" # "tool0"
manip_info.manipulator = "manipulator"
manip_info.working_frame = "base_link"

# Create a viewer and set the environment so the results can be displayed later
viewer = TesseractViewer()
# Show the world coordinate frame
viewer.add_axes_marker(position=[0,0,0], quaternion=[1,0,0,0], size=1.0, parent_link="base_link", name="world_frame")


viewer.update_environment(env, [0,0,0])

# Set the initial state of the robot
joint_names = ["cartesian_x_joint",
               "cartesian_y_joint",
               "cartesian_z_joint",
               "spherical_yaw_joint",
               "spherical_pitch_joint",
               "spherical_roll_joint",
               "cylindrical_1_to_cylindrical_2"]

# Set the initial state of the robot
init_x = 0
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
goal_x = 0
goal_y = -1.0
goal_z = 0.16

goal_yaw = np.pi/2
goal_pitch = 0
goal_roll = np.pi/2

goal_q = tf_trans.quaternion_from_euler(*[goal_roll, goal_pitch, goal_yaw], 'sxyz') # in xyzw format!!
goal_q = [goal_q[3], goal_q[0], goal_q[1], goal_q[2]] # convert to wxyz format

# Add the goal pose to the viewer
viewer.add_axes_marker(position=[goal_x,goal_y,goal_z], quaternion=goal_q, size=0.5, parent_link="base_link", name="goal_frame")

initial_joint_positions = np.array([init_x, init_y, init_z, init_yaw, init_pitch, init_roll, 0.0])

viewer.update_joint_positions(joint_names, initial_joint_positions)

# Start the viewer
viewer.start_serve_background()

# add_environment_obstacles(env, viewer, obstacle_ceiling_active=False)

# Set the initial state of the robot
env.setState(joint_names, np.ones(6)*0.1)

# Create the input command program waypoints
wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,-0.3,1.455) * Quaterniond(0.70710678,0,0.70710678,0))
wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,0.3,1.455) * Quaterniond(0.70710678,0,0.70710678,0))

# Create the input command program instructions. Note the use of explicit construction of the CartesianWaypointPoly
# using the *_wrap_CartesianWaypoint functions. This is required because the Python bindings do not support implicit
# conversion from the CartesianWaypoint to the CartesianWaypointPoly.
start_instruction = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
plan_f1 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")

# Create the input command program. Note the use of *_wrap_MoveInstruction functions. This is required because the
# Python bindings do not support implicit conversion from the MoveInstruction to the MoveInstructionPoly.
program = CompositeInstruction("DEFAULT")
program.setManipulatorInfo(manip_info)
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
# program.appendMoveInstruction(MoveInstructionPoly(plan_f2))


# -----------------------------------------------------------------------------------------------
# Initialize the OMPL planner for RRTConnect algorithm
plan_profile = OMPLDefaultPlanProfile()
plan_profile.planners.clear()
plan_profile.planners.append(RRTConnectConfigurator())

# Create the profile dictionary. Profiles can be used to customize the behavior of the planner. The module
# level function `ProfileDictionary_addProfile_OMPLPlanProfile` is used to add a profile to the dictionary. All
# profile types have associated profile dictionary functions.
profiles = ProfileDictionary()
ProfileDictionary_addProfile_OMPLPlanProfile(profiles,OMPL_DEFAULT_NAMESPACE, "TEST_PROFILE", plan_profile)

cur_state = env.getState()

# Create the planning request and run the planner
request = PlannerRequest()
request.instructions = program
request.env = env
request.env_state = cur_state
request.profiles = profiles

# Create the OMPL planner
ompl_planner = OMPLMotionPlanner(OMPL_DEFAULT_NAMESPACE) 

response=ompl_planner.solve(request)
assert response.successful
results_instruction = response.results

# The OMPL program does not generate dense waypoints. This function will interpolate the results to generate
# a dense set of waypoints.
interpolated_results_instruction = generateInterpolatedProgram(results_instruction, cur_state, env, 3.14, 1.0, 3.14, 10)

# -----------------------------------------------------------------------------------------------
# Create the TrajOpt planner profile configurations. TrajOpt is used to optimize the random program generated
# by OMPL
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

trajopt_profiles = ProfileDictionary()
ProfileDictionary_addProfile_TrajOptPlanProfile(trajopt_profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_plan_profile)
ProfileDictionary_addProfile_TrajOptCompositeProfile(trajopt_profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_composite_profile)


# Create the TrajOpt planning request and run the planner
trajopt_request = PlannerRequest()
trajopt_request.instructions = interpolated_results_instruction
trajopt_request.env = env
trajopt_request.env_state = cur_state
trajopt_request.profiles = trajopt_profiles

# Create the TrajOpt planner
trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

trajopt_response = trajopt_planner.solve(trajopt_request)
assert trajopt_response.successful
    
trajopt_results_instruction =trajopt_response.results
# -----------------------------------------------------------------------------------------------

# The TrajOpt planner does not assign timestamps to the results. This function will assign timestamps to the
# results using the TimeOptimalTrajectoryGeneration class. This class uses the velocity and acceleration limits
# to compute timestamps for the results. The input program is modified to assign timestamps, so there is no
# output program since the input is modified.
time_parameterization = TimeOptimalTrajectoryGeneration()
instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)
max_velocity = np.array([2.088, 2.082, 3.27, 3.6, 3.3, 3.078],dtype=np.float64)
max_acceleration = np.array([ 1, 1, 1, 1, 1, 1],dtype=np.float64)
assert time_parameterization.computeTimeStamps(instructions_trajectory, max_velocity, max_acceleration)

# Flatten the results into a single list of instructions
trajopt_results = trajopt_results_instruction.flatten()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Print out the resulting waypoints
for instr in trajopt_results:
    assert instr.isMoveInstruction()
    move_instr1 = InstructionPoly_as_MoveInstructionPoly(instr)
    wp1 = move_instr1.getWaypoint()
    assert wp1.isStateWaypoint()
    wp = WaypointPoly_as_StateWaypointPoly(wp1)
    print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")

# Update the viewer with the results to animate the trajectory
# Open web browser to http://localhost:8000 to view the results
viewer.update_trajectory(trajopt_results)
viewer.plot_trajectory(trajopt_results, manip_info, axes_length=0.05)

input("press enter to exit")

