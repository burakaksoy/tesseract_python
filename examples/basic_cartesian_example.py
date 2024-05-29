import os
import re
import traceback
import numpy as np

import time
import sys
from pathlib import Path

import tesseract_robotics.tesseract_common as tesseract_common
import tesseract_robotics.tesseract_environment as tesseract_environment
import tesseract_robotics.tesseract_scene_graph as tesseract_scene_graph
import tesseract_robotics.tesseract_geometry as tesseract_geometry
import tesseract_robotics.tesseract_command_language as tesseract_command_language
import tesseract_robotics.tesseract_task_composer as tesseract_task_composer

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


from tesseract_robotics.tesseract_task_composer import  TaskComposerPluginFactory, \
                                                        PlanningTaskComposerProblem, \
                                                        PlanningTaskComposerProblemUPtr, \
                                                        PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr, \
                                                        TaskComposerDataStorage, \
                                                        TaskComposerContext, \
                                                        TaskComposerFuture, \
                                                        TaskComposerFutureUPtr




TESSERACT_SUPPORT_DIR = Path(os.environ["TESSERACT_RESOURCE_PATH"])
TESSERACT_TASK_COMPOSER_DIR = Path(os.environ["TESSERACT_TASK_COMPOSER_DIR"])
TASK_COMPOSER_PLUGIN_YAML = Path(r"config/task_composer_plugins.yaml")
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"


def get_environment(url) -> Environment:
    """
    given a `url` load a URDF & SRDF and return an Enviornment and Manipulator instance and a
    list of joint names
    """
    locator = GeneralResourceLocator()
    env = Environment()
    # tesseract_support = os.environ["TESSERACT_SUPPORT_DIR"]
    urdf_path = locator.locateResource(f"{url}.urdf").getFilePath()
    srdf_path = locator.locateResource(f"{url}.srdf").getFilePath()

    urdf_path_str = FilesystemPath(urdf_path)
    srdf_path_str = FilesystemPath(srdf_path)

    assert env.init(urdf_path_str, srdf_path_str, locator)

    return env


class BasicCartesianExample:
    def __init__(self, env: Environment, visualize_results: bool = True, ifopt: bool = False, debug: bool = False):
        self.env = env
        self.visualize_results = visualize_results
        self.ifopt = ifopt
        self.debug = debug

        if self.visualize_results:
            self.viewer = TesseractViewer()
            self.viewer.update_environment(self.env, [0, 0, 0])
            self.viewer.start_serve_background()



    def add_point_cloud(self) -> AddLinkCommand:
        """Create octomap and add it to the local environment"""

        visual = Visual()
        visual.origin = Isometry3d.Identity()
        # As a default set dtype to np.float64 for numpy arrays
        # https://github.com/tesseract-robotics/tesseract_python/issues/55
        visual.origin.setTranslation(np.array([1, 0, 0], dtype=np.float64))

        visual.geometry = Box(1, 1, 1) 

        collision = Collision()
        collision.origin = visual.origin
        collision.geometry = visual.geometry

        link_box = Link("octomap_attached")
        link_box.visual.push_back(visual)
        link_box.collision.push_back(collision)

        joint = Joint("joint_octomap_attached")
        joint.parent_link_name = "base_link"
        joint.child_link_name = link_box.getName()
        joint.type = JointType_FIXED

        cmd = AddLinkCommand(link_box, joint)
        self.env.applyCommand(cmd)

        if self.visualize_results:
            self.viewer.update_environment(self.env, [0, 0, 0])

        return cmd

    def run(self):
        # Create octomap and add it to the local environment
        cmd = self.add_point_cloud()

        # Set the robot initial state
        joint_names = ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "joint_a7"]
        joint_position = np.array([-0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0])

        if self.visualize_results:
            self.viewer.update_joint_positions(joint_names, joint_position)

        if not self.env.isInitialized():
            # Prevent segmentation fault
            # https://github.com/tesseract-robotics/tesseract_python/issues/54
            raise RuntimeError("Environment not initialized")
        self.env.setState(joint_names, joint_position)

        if self.debug:
            tesseract_common.setLogLevel(CONSOLE_BRIDGE_LOG_DEBUG)


        # Create Task Composer Plugin Factory
        task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]
        config_path = FilesystemPath(task_composer_filename)
        factory = TaskComposerPluginFactory(config_path)


        # Create program
        # Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
        # match the SRDF, although the exact tcp_frame can differ if a tool is used.
        manip_info = ManipulatorInfo()
        manip_info.tcp_frame = "tool0"
        manip_info.manipulator = "manipulator"
        manip_info.working_frame = "base_link"

        program = CompositeInstruction(
            "cartesian_program", CompositeInstructionOrder_ORDERED, manip_info
        )

        # Start Joint Position for the program
        wp0 = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(joint_names, joint_position))
        start_instruction = MoveInstruction(
            wp0, MoveInstructionType_FREESPACE, "freespace_profile"
        )
        start_instruction.setDescription("Start Instruction")

        # Create cartesian waypoint
        wp1 = CartesianWaypointPoly_wrap_CartesianWaypoint(
            CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.5, -0.2, 0.62) * Quaterniond(0, 0, 1.0, 0))
        )
        wp2 = CartesianWaypointPoly_wrap_CartesianWaypoint(
            CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.5, 0.3, 0.62) * Quaterniond(0, 0, 1.0, 0))
        )

        # Plan freespace from start
        plan_f0 = MoveInstruction(wp1, MoveInstructionType_FREESPACE, "freespace_profile")
        plan_f0.setDescription("from_start_plan")

        # Plan linear move
        plan_c0 = MoveInstruction(wp2, MoveInstructionType_LINEAR, "RASTER")

        # Plan freespace to end
        plan_f1 = MoveInstruction(wp0, MoveInstructionType_FREESPACE, "freespace_profile")
        plan_f1.setDescription("to_end_plan")

        # Add instructions to program
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f0))
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_c0))
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
        

        # Print diagnosics
        program._print("Program: ")
        # CONSOLE_BRIDGE_logInform("basic cartesian plan example")

        # Create executor
        executor = factory.createTaskComposerExecutor("TaskflowExecutor")

        # --------------------------------------------------------------------------------------------
        # Create profile dictionary
        profiles = ProfileDictionary()

        if self.ifopt:
            # I am only able to find the base, descartes, ompl, simple, and trajopt motion planners under
            # tesseract_robotics.tesseract_motion_planners_*. It seems trajopt_ifopt has not been ported yet.
            raise NotImplementedError("TrajOptIfopt has not been ported to Python yet.")

        else:
            composite_profile = TrajOptDefaultCompositeProfile()
            composite_profile.collision_cost_config.enabled = True
            composite_profile.collision_constraint_config.enabled = True
            composite_profile.smooth_velocities = True
            composite_profile.smooth_accelerations = False
            composite_profile.smooth_jerks = False
            composite_profile.velocity_coeff = np.array([1], dtype=np.float64)
            ProfileDictionary_addProfile_TrajOptCompositeProfile(
                profile_dictionary=profiles,
                ns=TRAJOPT_DEFAULT_NAMESPACE,
                profile_name="cartesian_program",
                profile=composite_profile,
            )

            plan_profile = TrajOptDefaultPlanProfile()
            plan_profile.cartesian_coeff = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
            plan_profile.joint_coeff = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
            ProfileDictionary_addProfile_TrajOptPlanProfile(
                profile_dictionary=profiles,
                ns=TRAJOPT_DEFAULT_NAMESPACE,
                profile_name="RASTER",
                profile=plan_profile,
            )
            ProfileDictionary_addProfile_TrajOptPlanProfile(
                profile_dictionary=profiles,
                ns=TRAJOPT_DEFAULT_NAMESPACE,
                profile_name="freespace_profile",
                profile=plan_profile,
            )
        # --------------------------------------------------------------------------------------------

        # Create task
        task_name = "TrajOptIfoptPipeline" if self.ifopt else "TrajOptPipeline"
        # The next line will create and error if the loaded config is incorrect: https://github.com/tesseract-robotics/tesseract_python/issues/52
        task = factory.createTaskComposerNode(task_name)
        input_key = task.getInputKeys()[0]
        output_key = task.getOutputKeys()[0]

        # Create Task Composer problem
        task_data = TaskComposerDataStorage()
        task_data.setData(input_key, AnyPoly_wrap_CompositeInstruction(program))
        problem = PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr(PlanningTaskComposerProblemUPtr.make_unique(self.env, task_data, profiles))
        task_composer_input = TaskComposerInput(problem)

        if self.visualize_results:
            input("Hit Enter to solve for trajectory")

        # Solve task
        stopwatch = Timer()
        stopwatch.start()
        future = executor.run(task.get(), task_composer_input)
        future.wait()

        stopwatch.stop()
        print(f"Planning took {stopwatch.elapsedSeconds()} seconds.")

        # Plot Process Trajectory
        if self.visualize_results:
            # TaskComposerFuture does not expose the `context` so have to work around this
            composite_instuction = AnyPoly_as_CompositeInstruction(task_data.getData(output_key))
            toolpath = toToolpath(composite_instuction, self.env)
            trajectory = toJointTrajectory(composite_instuction)
            # state_solver = self.env.getStateSolver()
            # plotMarker(ToolpathMarger(toolpath))
            # plotTrajectory(trajectory, state_solver)

        print("Final trjectory is collision free")
        # return future.context.isSuccessful() # Context is not shared


def main():
    env = get_environment("package://tesseract_support/urdf/lbr_iiwa_14_r820")
    x = BasicCartesianExample(env, debug=True)
    x.run()

    input("Press enter to exit")


if __name__ == "__main__":
    main()
