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

from tesseract_robotics.tesseract_task_composer import  TaskComposerPluginFactory, \
                                                        PlanningTaskComposerProblem, \
                                                        PlanningTaskComposerProblemUPtr, \
                                                        PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr, \
                                                        TaskComposerDataStorage, \
                                                        TaskComposerContext, \
                                                        TaskComposerFuture, \
                                                        TaskComposerFutureUPtr

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



OMPL_DEFAULT_NAMESPACE = "OMPLMotionPlannerTask"
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"

task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]

# Initialize the resource locator and environment
locator = GeneralResourceLocator()
urdf_package_url = "package://tesseract_support/urdf/lbr_iiwa_14_r820.urdf"
srdf_package_url = "package://tesseract_support/urdf/lbr_iiwa_14_r820.srdf"
urdf_fname = FilesystemPath(locator.locateResource(urdf_package_url).getFilePath())
srdf_fname = FilesystemPath(locator.locateResource(srdf_package_url).getFilePath())

env = Environment()

# locator_fn must be kept alive by maintaining a reference
assert env.init(urdf_fname, srdf_fname, locator)

# Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
# match the SRDF, although the exact tcp_frame can differ if a tool is used.
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "tool0"
manip_info.manipulator = "manipulator"
manip_info.working_frame = "base_link"
# manip_info = ManipulatorInfo("manipulator", "tool0", "base_link")

# Create a viewer and set the environment so the results can be displayed later
viewer = TesseractViewer()
viewer.update_environment(env, [0,0,0])

# Set the initial state of the robot
joint_names = ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "joint_a7"]
joint_position = np.array([-0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0])
initial_joint_position = np.ones(7)*0.1

viewer.update_joint_positions(joint_names, initial_joint_position)

# Start the viewer
viewer.start_serve_background()

# # Add a sphere using Environment commands
# sphere_link = Link("sphere_link")
# sphere_link_visual = Visual()
# sphere_link_visual.geometry = Sphere(0.1)
# sphere_link.visual.push_back(sphere_link_visual)
# sphere_link_collision = Collision()
# sphere_link_collision.geometry = Sphere(0.1)
# sphere_link.collision.push_back(sphere_link_collision)
# sphere_joint = Joint("sphere_joint")
# sphere_joint.parent_link_name = "base_link"
# sphere_joint.child_link_name = sphere_link.getName()
# sphere_joint.type = JointType_FIXED
# sphere_link_joint_transform = Isometry3d.Identity() * Translation3d(0.7, 0, 1.5)
# sphere_joint.parent_to_joint_origin_transform = sphere_link_joint_transform
# add_sphere_command = AddLinkCommand(sphere_link, sphere_joint)
# env.applyCommand(add_sphere_command)

# viewer.update_environment(env, [0,0,0])

# Set the initial state of the robot
env.setState(joint_names, initial_joint_position)

# time.sleep(5)

# ----------------------------------------------------------------------------------------------------------
# """

program = CompositeInstruction("cartesian_program", CompositeInstructionOrder_ORDERED, manip_info)


# Create the input command program waypoints
wp0 = StateWaypoint(joint_names, initial_joint_position)
start_instruction = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp0), MoveInstructionType_FREESPACE, "freespace_profile")
start_instruction.setDescription("Start Instruction")

wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.5, -0.2, 0.62) * Quaterniond(0, 0, 1.0, 0))

wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.5, 0.3, 0.62) * Quaterniond(0, 0, 1.0, 0))

# Create the input command program instructions. Note the use of explicit construction of the CartesianWaypointPoly
# using the *_wrap_CartesianWaypoint functions. This is required because the Python bindings do not support implicit
# conversion from the CartesianWaypoint to the CartesianWaypointPoly.

# Plan linear move
plan_c0 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_LINEAR, "RASTER") # "DEFAULT")


plan_f0 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "freespace_profile")
plan_f0.setDescription("from_start_plan")


from_start = CompositeInstruction()
from_start.setManipulatorInfo(manip_info)
from_start.setDescription("from_start")
from_start.
from_start.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
program.append(from_start)
# from_start.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f0))

# raster_segment = CompositeInstruction()
# raster_segment.setManipulatorInfo(manip_info)
# raster_segment.setDescription("raster_segment")
# raster_segment.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_c0))
# program.append(raster_segment)


# # Plan free space to end
plan_f1 = MoveInstruction(StateWaypointPoly_wrap_StateWaypoint(wp0), MoveInstructionType_FREESPACE, "freespace_profile") 
plan_f1.setDescription("transition_from_end_plan")

# transition_from_end = CompositeInstruction()
# transition_from_end.setManipulatorInfo(manip_info)
# transition_from_end.setDescription("transition_from_end")
# transition_from_end.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
# program.append(transition_from_end)


program.appendMoveInstruction(plan_f1)

# Print diagnosics
program._print("Program: ")

# Create an AnyPoly containing the program. This explicit step is required because the Python bindings do not
# support implicit conversion from the CompositeInstruction to the AnyPoly.
program_anypoly = AnyPoly_wrap_CompositeInstruction(program)

# Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
# in the instructions.
profiles = ProfileDictionary()
# Create the task problem and input
task_planning_problem = PlanningTaskComposerProblem(env, profiles)
task_planning_problem.input = program_anypoly

# Create the task composer plugin factory and load the plugins
config_path = FilesystemPath(task_composer_filename)
factory = TaskComposerPluginFactory(config_path)

# Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
task = factory.createTaskComposerNode("FreespacePipeline")
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
    print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")

# Update the viewer with the results to animate the trajectory
# Open web browser to http://localhost:8000 to view the results
viewer.update_trajectory(results)
viewer.plot_trajectory(results, manip_info)

# """
# ----------------------------------------------------------------------------------------------------------

input("press enter to exit")