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
                                                               
from tesseract_robotics.tesseract_urdf import parseURDFString, \
                                              parseURDFFile, \
                                              writeURDFFile
                                                        

import tf.transformations as tf_trans

from examples_utils.add_env_obstacles import add_environment_obstacles, \
                                             add_environment_obstacles_l_shape_corridor, \
                                             add_environment_obstacles_from_urdf
                                             
from examples_utils.add_profiles import add_MinLengthProfile, \
                                        add_TrajOptPlanProfile, \
                                        add_OMPLDefaultPlanProfile

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

setLogLevel(CONSOLE_BRIDGE_LOG_DEBUG)

task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]

# Initialize the resource locator and environment
locator = GeneralResourceLocator()
urdf_package_url = "package://tesseract_support/urdf/deformable_description/urdf/pole_2_links_7DOF/pole_2_links.urdf"
srdf_package_url = "package://tesseract_support/urdf/deformable_description/urdf/pole_2_links_7DOF/pole_2_links.srdf"
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



# Set the initial state of the robot
pos_x = 0 # -3.3528/2.0 # 0

init_x = pos_x # 0
init_y = 2.5 # 2.0
init_z = 0.5

init_yaw = np.pi/2
init_pitch = 0
init_roll = np.pi/2

init_q = tf_trans.quaternion_from_euler(*[init_roll, init_pitch, init_yaw], 'sxyz') # in xyzw format!!
init_q = [init_q[3], init_q[0], init_q[1], init_q[2]] # convert to wxyz format

# Add the initial pose to the viewer
viewer.add_axes_marker(position=[init_x,init_y,init_z], quaternion=init_q, size=0.5, parent_link="base_link", name="init_frame")

# Set the goal state of the robot
goal_x = 2.5 # 0.0
goal_y = 0.0 # -1.0
goal_z = 0.5 # 0.16

goal_yaw = 0.0 # np.pi/2
goal_pitch = 0.0
goal_roll = np.pi/2.0

goal_q = tf_trans.quaternion_from_euler(*[goal_roll, goal_pitch, goal_yaw], 'sxyz') # in xyzw format!!
goal_q = [goal_q[3], goal_q[0], goal_q[1], goal_q[2]] # convert to wxyz format

# Add the goal pose to the viewer
viewer.add_axes_marker(position=[goal_x,goal_y,goal_z], quaternion=goal_q, size=0.5, parent_link="base_link", name="goal_frame")



# initial_joint_positions = np.array([init_x, init_y, init_z, init_yaw, init_pitch, init_roll])
initial_joint_positions = np.array([init_x, init_y, init_z, init_yaw, init_pitch, init_roll, 0.0])

viewer.update_joint_positions(joint_names, initial_joint_positions)

# Start the viewer
viewer.start_serve_background()

# --------------------------------------------------------------------------------------------
# add_environment_obstacles(env, viewer, obstacle_ceiling_active=False)
# add_environment_obstacles(env, viewer, obstacle_ceiling_active=True)

urdf_path = "/home/burak/tesseract_learning/tesseract/tesseract_support/urdf/deformable_description/urdf/urdf_exported/"
# urdf_path = os.path.join(os.environ["TESSERACT_RESOURCE_PATH"], "tesseract_support/urdf/deformable_description/urdf/urdf_exported/urdf/")
urdf_name = "l_shape_corridor" # .urdf extension is added in the function automatically

add_environment_obstacles_l_shape_corridor(env, viewer, ceiling_active=True,
                                           root_link_name="corridor_base_link",
                                           parent_link_name="base_link",
                                           export_as_urdf=False, 
                                           urdf_path=urdf_path,
                                           urdf_name=urdf_name)


# add_environment_obstacles_from_urdf(env, viewer, 
#                                     urdf_path=urdf_path + "urdf/",
#                                     urdf_name = urdf_name + ".urdf")
# --------------------------------------------------------------------------------------------



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

# env.getContinuousContactManager()
# env.getDiscreteContactManager()



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
add_MinLengthProfile(profiles, "DEFAULT")

add_TrajOptPlanProfile(profiles, "DEFAULT")

add_OMPLDefaultPlanProfile(profiles, "DEFAULT")
# --------------------------------------------------------------------------------------------

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

# # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation
# task = factory.createTaskComposerNode("FreespacePipeline2") 
# # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation
# task = factory.createTaskComposerNode("TrajOptPipeline2") 

# Get the output keys for the task
output_key = task.getOutputKeys()[0]

# Create an executor to run the task
task_executor = factory.createTaskComposerExecutor("TaskflowExecutor")

# --------------------------------------------------------------------------------------------
# Solve task
stopwatch = Timer()
stopwatch.start()

# Run the task and wait for completion
future = task_executor.run(task.get(), task_planning_problem)
future.wait()

stopwatch.stop()
print(f"PLANNING TOOK {stopwatch.elapsedSeconds()} SECONDS.")
# --------------------------------------------------------------------------------------------

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