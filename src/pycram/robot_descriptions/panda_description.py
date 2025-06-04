import math
import os

import pathlib
from ..tf_transformations import quaternion_from_euler

from ..datastructures.dataclasses import ManipulatorData
from ..datastructures.enums import Grasp, Arms
from ..helper import get_robot_urdf_and_mjcf_file_paths, find_multiverse_resources_path
from ..robot_description import RobotDescriptionManager, create_manipulator_description
from ..ros import logwarn, loginfo
from ..units import meter
from ..config.world_conf import WorldConfig

data = ManipulatorData(
    name="panda",
    relative_dir="franka_emika_panda",
    base_link="link0",

    arm_end_link="link7",
    joint_names=[f'joint{i}' for i in range(1, 8)],
    home_joint_values=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853],

    gripper_name="hand",
    gripper_start_link="hand",
    gripper_relative_dir=None,
    gripper_tool_frame="gripper_tool_frame",

    gripper_joint_names=[f'finger_joint{i}' for i in [1, 2]],
    closed_joint_values=[0.0, 0.0],
    open_joint_values=[0.04, 0.04],
    opening_distance=0.08 * meter,
    fingers_link_names=["left_finger", "right_finger"])

multiverse_resources = find_multiverse_resources_path()

urdf_filename, mjcf_filename = None, None
if multiverse_resources is not None:
    urdf_filename, mjcf_filename = get_robot_urdf_and_mjcf_file_paths(data.name, data.relative_dir,
                                                                      multiverse_resources=multiverse_resources)
if urdf_filename is None and mjcf_filename is not None:
    from multiverse_parser import MjcfImporter, UrdfExporter
    factory = MjcfImporter(file_path=mjcf_filename,
                           fixed_base=True,
                           root_name=data.base_link,
                           with_visual=True,
                           with_collision=True,
                           with_physics=True)
    factory.import_model()
    filename_from_path = os.path.basename(mjcf_filename).split(".")[0]
    urdf_filename = os.path.join(multiverse_resources, "cached", f"{filename_from_path}.urdf")
    mjcf_exporter = UrdfExporter(factory=factory, file_path=str(urdf_filename))
    mjcf_exporter.build()
    mjcf_exporter.export(keep_usd=False)

if mjcf_filename is None and urdf_filename is None:
    logwarn(f"Could not initialize {data.name} description as Multiverse resources path not found.")
else:
    robot_description = create_manipulator_description(data, urdf_filename, mjcf_filename)
    ################################# Grasps ##################################
    robot_description.get_arm_chain(Arms.RIGHT).end_effector.update_all_grasp_orientations([0, 0, 0, 1])

    # Add to RobotDescriptionManager
    rdm = RobotDescriptionManager()
    rdm.register_description(robot_description)
