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
                                                          CompositeInstructionOrder_UNORDERED, \
                                                          CompositeInstructionOrder_ORDERED_AND_REVERABLE, \
                                                          DEFAULT_PROFILE_KEY, \
                                                          JointWaypoint, \
                                                          JointWaypointPoly, \
                                                          StateWaypointPoly_wrap_StateWaypoint, \
                                                          JointWaypointPoly_wrap_JointWaypoint, \
                                                          toJointTrajectory

# CompositeInstructionOrder_ORDERED: Must go in forward
# CompositeInstructionOrder_UNORDERED: Any order is allowed
# CompositeInstructionOrder_ORDERED_AND_REVERABLE: Can go forward or reverse the order

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
abb_irb2400_urdf_package_url = "package://tesseract_support/urdf/abb_irb2400.urdf"
abb_irb2400_srdf_package_url = "package://tesseract_support/urdf/abb_irb2400.srdf"
abb_irb2400_urdf_fname = FilesystemPath(locator.locateResource(abb_irb2400_urdf_package_url).getFilePath())
abb_irb2400_srdf_fname = FilesystemPath(locator.locateResource(abb_irb2400_srdf_package_url).getFilePath())

env = Environment()

# locator_fn must be kept alive by maintaining a reference
assert env.init(abb_irb2400_urdf_fname, abb_irb2400_srdf_fname, locator)

# Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
# match the SRDF, although the exact tcp_frame can differ if a tool is used.
manip_info = ManipulatorInfo()
manip_info.manipulator = "manipulator"
manip_info.tcp_frame = "tool0"
manip_info.working_frame = "base_link"
# manip_info = ManipulatorInfo("manipulator", "tool0", "base_link")

# Create a viewer and set the environment so the results can be displayed later
viewer = TesseractViewer()
viewer.update_environment(env, [0,0,0])

# Set the initial state of the robot
joint_names = ["joint_%d" % (i+1) for i in range(6)]
viewer.update_joint_positions(joint_names, np.array([1,-.2,.01,.3,-.5,1]))

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
env.setState(joint_names, np.ones(6)*0.1)

program = CompositeInstruction("cartesian_program", CompositeInstructionOrder_ORDERED, manip_info)

# Create the input command program waypoints
wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,-0.3,1.455) * Quaterniond(0.70710678,0,0.70710678,0))

wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,0.3,1.455) * Quaterniond(0.70710678,0,0.70710678,0))

wp3 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,0.4,1.455) * Quaterniond(0.70710678,0,0.70710678,0))
# wp4 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.8,0.5,1.455) * Quaterniond(0.70710678,0,0.70710678,0))

# Create the input command program instructions. Note the use of explicit construction of the CartesianWaypointPoly
# using the *_wrap_CartesianWaypoint functions. This is required because the Python bindings do not support implicit
# conversion from the CartesianWaypoint to the CartesianWaypointPoly.
start_instruction = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
plan_f1 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")
plan_f2 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp3), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_f3 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp4), MoveInstructionType_FREESPACE, "DEFAULT")


#   CompositeInstruction(std::string profile = DEFAULT_PROFILE_KEY,
#                        CompositeInstructionOrder order = CompositeInstructionOrder::ORDERED,
#                        tesseract_common::ManipulatorInfo manipulator_info = tesseract_common::ManipulatorInfo());

# program = CompositeInstruction("DEFAULT")
# program = CompositeInstruction()
# program.setManipulatorInfo(manip_info)
# program.setProfileKey("DEFAULT")

# Create the input command program. Note the use of *_wrap_MoveInstruction functions. This is required because the
# Python bindings do not support implicit conversion from the MoveInstruction to the MoveInstructionPoly.
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
# program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f2))
# program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f3))





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
# task = factory.createTaskComposerNode("TrajOptPipeline")
# task = factory.createTaskComposerNode("OMPLPipeline")
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

input("press enter to exit")