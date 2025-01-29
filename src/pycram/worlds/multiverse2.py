import os

import mujoco
import numpy as np
from mujoco_connector.src.mujoco_connector import MultiverseMujocoConnector
from multiverse_simulator import MultiverseViewer, MultiverseAttribute
from typing_extensions import List, Optional, Dict, Callable, Type, Tuple

import pycrap
from pycrap import PhysicalObject
from ..config.multiverse_conf import MultiverseConfig
from ..datastructures.dataclasses import Color, ContactPointsList
from ..datastructures.enums import WorldMode, JointType, MultiverseJointCMD, MultiverseJointPosition
from ..datastructures.pose import Pose
from ..datastructures.world import World
from ..datastructures.world_entity import PhysicalBody
from ..description import Link, Joint
from ..object_descriptors.generic import ObjectDescription as GenericObjectDescription
from ..object_descriptors.mjcf import ObjectDescription as MJCF, PrimitiveObjectFactory
from ..robot_description import RobotDescription
from ..ros.logging import logwarn, logerr
from ..utils import RayTestUtils, xyzw_to_wxyz_arr, wxyz_to_xyzw, adjust_camera_pose_based_on_target
from ..world_concepts.constraints import Constraint
from ..world_concepts.world_object import Object


class Multiverse(World):
    """
    This class implements an interface between Multiverse and PyCRAM.
    """
    conf: MultiverseConfig = MultiverseConfig
    """
    The Multiverse configuration.
    """

    supported_joint_types = (JointType.REVOLUTE, JointType.CONTINUOUS, JointType.PRISMATIC)
    """
    A Tuple for the supported pycram joint types in Multiverse.
    """

    added_multiverse_resources: bool = False
    """
    A flag to check if the multiverse resources have been added.
    """

    Object.extension_to_description_type[MJCF.get_file_extension()] = MJCF
    """
    Add the MJCF description extension to the extension to description type mapping for the objects.
    """

    def _init_world(self, mode: WorldMode):
        pass

    def __init__(self, mode: WorldMode = WorldMode.DIRECT,
                 is_prospection: Optional[bool] = False,
                 clear_cache: bool = False,
                 prospection_mode: WorldMode = WorldMode.DIRECT,
                 scene_file_path: Optional[str] = None):
        """
        Initialize the Multiverse Socket and the PyCram World.

        :param is_prospection: Whether the world is prospection or not.
        :param clear_cache: Whether to clear the cache or not.
        :param prospection_mode: The mode of the prospection world.
        :param scene_file_path: The path to the scene file that needs to be loaded in the simulator.
        """

        self.latest_save_id: Optional[int] = None
        self.saved_simulator_states: Dict = {}
        self.make_sure_multiverse_resources_are_added(clear_cache=clear_cache)

        self.simulation = self.conf.prospection_world_prefix if is_prospection else "belief_state"

        if scene_file_path is None:
            scene_file_path = os.path.join(self.conf.resources_path,
                                           "worlds/floor/floor.xml")
        self._scene_file_path = scene_file_path

        World.__init__(self, mode=mode, is_prospection=is_prospection, prospection_mode=prospection_mode,
                       scene_file_path=scene_file_path)

        self._init_constraint_and_object_id_name_map_collections()

        self.ray_test_utils = RayTestUtils(self.ray_test_batch, self.object_id_to_name)

        self.viewer = MultiverseViewer()
        self._init_scene_and_update_viewer()
        self.simulator = MultiverseMujocoConnector(viewer=self.viewer,
                                                   file_path=scene_file_path,
                                                   headless=mode == WorldMode.DIRECT,
                                                   real_time_factor=1,
                                                   step_size=self.conf.simulation_time_step.total_seconds())
        self.simulator.start(run_in_thread=False)

        if not self.is_prospection_world:
            self._spawn_floor()

    def _init_scene_and_update_viewer(self) -> None:
        """
        Initialize the scene using the scene file path.
        """
        mj_model = mujoco.MjModel.from_xml_path(self.scene_file_path)
        read_objects = {}
        revolute_joint_names = [mj_model.joint(joint_id).name for joint_id in range(mj_model.njnt) if
                                mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE]
        for joint_name in revolute_joint_names:
            read_objects[joint_name] = {
                "joint_rvalue": [0.0]
            }
        prismatic_joint_names = [mj_model.joint(joint_id).name for joint_id in range(mj_model.njnt) if
                                 mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE]
        for joint_name in prismatic_joint_names:
            read_objects[joint_name] = {
                "joint_tvalue": [0.0]
            }
        self.viewer.read_objects = read_objects

    @property
    def scene_file_path(self) -> str:
        return self._scene_file_path

    @scene_file_path.setter
    def scene_file_path(self, value: str):
        self._scene_file_path = value

    @classmethod
    def make_sure_multiverse_resources_are_added(cls, clear_cache: bool = False) -> None:
        """
        Add the multiverse resources to the pycram world resources, and change the data directory and cache manager.

        :param clear_cache: Whether to clear the cache or not.
        """
        if not cls.added_multiverse_resources:
            if clear_cache:
                World.cache_manager.clear_cache()
            World.add_resource_path(cls.conf.resources_path, prepend=True)
            World.change_cache_dir_path(cls.conf.resources_path)
            cls.added_multiverse_resources = True

    def remove_multiverse_resources(self):
        """
        Remove the multiverse resources from the pycram world resources.
        """
        if self.added_multiverse_resources:
            World.remove_resource_path(self.conf.resources_path)
            World.change_cache_dir_path(self.conf.cache_dir)
            self.added_multiverse_resources = False

    def _init_constraint_and_object_id_name_map_collections(self):
        self.last_object_id: int = -1
        self.last_constraint_id: int = -1
        self.constraints: Dict[int, Constraint] = {}
        self.object_name_to_id: Dict[str, int] = {}
        self.object_id_to_name: Dict[int, str] = {}

    def _spawn_floor(self):
        """
        Spawn the plane in the simulator.
        """
        self.floor = Object("floor", pycrap.Floor, "plane.urdf",
                            world=self)

    def load_object_and_get_id(self, name: Optional[str] = None,
                               pose: Optional[Pose] = None,
                               obj_type: Optional[Type[PhysicalObject]] = None) -> int:
        """
        Spawn the object in the simulator and return the object id. Object name has to be unique and has to be same as
        the name of the object in the description file.

        :param name: The name of the object to be loaded.
        :param pose: The pose of the object.
        :param obj_type: The type of the object.
        """
        self._add_object_read_data_to_viewer(name)
        return self._update_object_id_name_maps_and_get_latest_id(name)

    def _add_object_read_data_to_viewer(self, name: str) -> None:
        """
        Add the object read data to the viewer.

        :param name: The name of the object.
        :param position: The position of the object.
        :param quaternion: The quaternion of the object.
        """
        read_objects = self.viewer.read_objects
        read_objects[name] = {
            "position": MultiverseAttribute(np.zeros(3)),
            "quaternion": MultiverseAttribute(np.array([1, 0, 0, 0]))
        }
        self.viewer.read_objects = read_objects

    def load_generic_object_and_get_id(self, description: GenericObjectDescription,
                                       pose: Optional[Pose] = None) -> int:
        # save_path = os.path.join(self.cache_manager.cache_dir, description.name + ".xml")
        # object_factory = PrimitiveObjectFactory(description.name, description.links[0].geometry, save_path)
        # object_factory.build_shape()
        # object_factory.export_to_mjcf(save_path)
        return self.load_object_and_get_id(description.name, pose, pycrap.PhysicalObject)

    def get_images_for_target(self, target_pose: Pose,
                              cam_pose: Pose,
                              size: int = 256,
                              camera_min_distance: float = 0.1,
                              camera_max_distance: int = 3,
                              plot: bool = False) -> List[np.ndarray]:
        """
        Uses ray test to get the images for the target object. (target_pose is currently not used)
        """
        camera_description = RobotDescription.current_robot_description.get_default_camera()
        camera_frame = RobotDescription.current_robot_description.get_camera_frame(World.robot.name)
        adjusted_cam_pose = adjust_camera_pose_based_on_target(cam_pose, target_pose, camera_description)
        return self.ray_test_utils.get_images_for_target(adjusted_cam_pose, camera_description, camera_frame,
                                                         size, camera_min_distance, camera_max_distance, plot)

    @staticmethod
    def get_joint_cmd_name(joint_type: JointType) -> MultiverseJointCMD:
        """
        Get the attribute name of the joint command in the Multiverse from the pycram joint type.

        :param joint_type: The pycram joint type.
        """
        return MultiverseJointCMD.from_pycram_joint_type(joint_type)

    def _update_object_id_name_maps_and_get_latest_id(self, name: str) -> int:
        """
        Update the object id name maps and return the latest object id.

        :param name: The name of the object.
        :return: The latest object id.
        """
        self.last_object_id += 1
        self.object_name_to_id[name] = self.last_object_id
        self.object_id_to_name[self.last_object_id] = name
        return self.last_object_id

    def _remove_visual_object(self, obj_id: int) -> bool:
        logwarn("Removing visual object is not supported in Multiverse.")
        return False

    def remove_object_from_simulator(self, obj: Object) -> bool:
        logwarn("Removing object from simulator is not supported in Multiverse.")
        return False

    def add_constraint(self, constraint: Constraint) -> int:

        if constraint.type != JointType.FIXED:
            logerr("Only fixed constraints are supported in Multiverse")
            raise ValueError

        if not self.conf.let_pycram_move_attached_objects:
            parent_link_name, child_link_name = self.get_constraint_link_names(constraint)
            attachment_pose = constraint.parent_to_child_transform.to_pose()
            self._attach(child_link_name, parent_link_name, attachment_pose)

        return self._update_constraint_collection_and_get_latest_id(constraint)

    def _attach(self, child_link_name: str, parent_link_name: str, attachment_pose: Pose) -> None:
        """
        Attach the child link to the parent link.

        :param child_link_name: The name of the child link.
        :param parent_link_name: The name of the parent link.
        :param attachment_pose: The attachment pose.
        """
        self.simulator.attach(child_link_name, parent_link_name, attachment_pose.position_as_array(),
                              xyzw_to_wxyz_arr(attachment_pose.orientation_as_array()))

    def _update_constraint_collection_and_get_latest_id(self, constraint: Constraint) -> int:
        """
        Update the constraint collection and return the latest constraint id.

        :param constraint: The constraint to be added.
        :return: The latest constraint id.
        """
        self.last_constraint_id += 1
        self.constraints[self.last_constraint_id] = constraint
        return self.last_constraint_id

    def get_constraint_link_names(self, constraint: Constraint) -> Tuple[str, str]:
        """
        Get the link names of the constraint.

        :param constraint: The constraint.
        :return: The link names of the constraint.
        """
        return self.get_parent_link_name(constraint), self.get_constraint_child_link_name(constraint)

    def get_parent_link_name(self, constraint: Constraint) -> str:
        """
        Get the parent link name of the constraint.

        :param constraint: The constraint.
        :return: The parent link name of the constraint.
        """
        return self.get_link_name_for_constraint(constraint.parent_link)

    def get_constraint_child_link_name(self, constraint: Constraint) -> str:
        """
        Get the child link name of the constraint.

        :param constraint: The constraint.
        :return: The child link name of the constraint.
        """
        return self.get_link_name_for_constraint(constraint.child_link)

    @staticmethod
    def get_link_name_for_constraint(link: Link) -> str:
        """
        Get the link name from link object, if the link belongs to a one link object, return the object name.

        :param link: The link.
        :return: The link name.
        """
        return link.name if not link.is_only_link else link.object.name

    def remove_constraint(self, constraint_id) -> None:
        constraint = self.constraints.pop(constraint_id)
        parent_link_name, child_link_name = self.get_constraint_link_names(constraint)
        self._detach(child_link_name)

    def _detach(self, child_link_name: str) -> None:
        """
        Detach the child link from the parent link.

        :param child_link_name: The name of the child link.
        """
        self.simulator.detach(child_link_name)

    def _get_joint_position(self, joint: Joint) -> float:
        joint_position_name = self.get_joint_position_name(joint)
        return self.viewer.read_objects[joint.name][joint_position_name.value].values[0][0]

    @staticmethod
    def get_joint_position_name(joint: Joint) -> MultiverseJointPosition:
        """
        Get the attribute name of the joint position in the Multiverse from the pycram joint type.

        :param joint: The joint.
        """
        return MultiverseJointPosition.from_pycram_joint_type(joint.type)

    def get_object_joint_names(self, obj: Object) -> List[str]:
        return [joint.name for joint in obj.description.joints if joint.type in self.supported_joint_types]

    def get_link_pose(self, link: Link) -> Pose:
        pos = self.get_link_position(link)
        orientation = self.get_link_orientation(link)
        return Pose(pos, orientation)

    def get_multiple_link_poses(self, links: List[Link]) -> Dict[str, Pose]:
        return {link.name: self.get_link_pose(link) for link in links}

    def get_link_position(self, link: Link) -> List[float]:
        if link.object.obj_type == pycrap.Floor:
            return [0, 0, 0]
        pos_arr = self.viewer.read_objects[link.name]["position"].values[0]
        return pos_arr.tolist()

    def get_link_orientation(self, link: Link) -> List[float]:
        if link.object.obj_type == pycrap.Floor:
            return [0, 0, 0, 1]
        quat_arr = self.viewer.read_objects[link.name]["quaternion"].values[0]
        return wxyz_to_xyzw(quat_arr.tolist())

    def get_multiple_link_positions(self, links: List[Link]) -> Dict[str, List[float]]:
        return {link.name: self.get_link_position(link) for link in links}

    def get_multiple_link_orientations(self, links: List[Link]) -> Dict[str, List[float]]:
        return {link.name: self.get_link_orientation(link) for link in links}

    def get_object_link_names(self, obj: Object) -> List[str]:
        return [link.name for link in obj.description.links]

    def get_object_pose(self, obj: Object) -> Pose:
        return self.get_link_pose(obj.root_link)

    def get_multiple_object_poses(self, objects: List[Object]) -> Dict[str, Pose]:
        return {obj.name: self.get_object_pose(obj) for obj in objects}

    def get_multiple_object_positions(self, objects: List[Object]) -> Dict[str, List[float]]:
        return {obj.name: self.get_object_position(obj) for obj in objects}

    def get_object_position(self, obj: Object) -> List[float]:
        return self.get_link_position(obj.root_link)

    def get_multiple_object_orientations(self, objects: List[Object]) -> Dict[str, List[float]]:
        return {obj.name: self.get_object_orientation(obj) for obj in objects}

    def get_object_orientation(self, obj: Object) -> List[float]:
        return self.get_link_orientation(obj.root_link)

    def perform_collision_detection(self) -> None:
        pass

    def get_body_contact_points(self, body: PhysicalBody) -> ContactPointsList:
        contacts = self.simulator.get_contact_points(body_1_name=body.name, including_children=True).result
        print(contacts)
        logwarn("multiverse contact points needs testing")
        return contacts

    def get_contact_points_between_two_bodies(self, body_1: PhysicalBody, body_2: PhysicalBody) -> ContactPointsList:
        contacts = self.simulator.get_contact_points(body_1_name=body_1.name, body_2_name=body_2.name,
                                                     including_children=True).result
        print(contacts)
        logwarn("multiverse contact points between two bodies needs testing")
        return contacts

    def _reset_joint_position(self, joint: Joint, joint_position: float) -> bool:
        if joint.name not in self.simulator.get_all_joint_names():
            logwarn(f"joint {joint.name} not found in the simulator.")
            return False
        write_objects = {
            joint.name: {
                self.get_joint_position_name(joint).value: [joint_position]
            }
        }
        self._write_and_step(write_objects)
        return True

    def _set_multiple_joint_positions(self, joint_positions: Dict[Joint, float]) -> bool:
        joints_data = {joint.name: {self.get_joint_position_name(joint): [position]}
                       for joint, position in joint_positions.items()
                       if joint.name in self.simulator.get_all_joint_names()}
        if len(joints_data) != len(joint_positions):
            logwarn("joint names not found in the simulator.")
            return False
        self._write_and_step(joints_data)
        return True

    def _write_and_step(self, write_objects: Dict[str, Dict]) -> None:
        """
        Write the objects data and step the simulator.
        """
        self.viewer.write_objects = write_objects
        self.simulator.step()

    def _get_multiple_joint_positions(self, joints: List[Joint]) -> Dict[str, float]:
        return {joint.name: self._get_joint_position(joint) for joint in joints}

    def reset_object_base_pose(self, obj: Object, pose: Pose) -> bool:
        if obj.name not in self.simulator.get_all_body_names().result:
            logwarn(f"object {obj.name} not found in the simulator.")
            return False
        write_objects = {
            obj.name: {
                "position": pose.position_as_array(),
                "quaternion": xyzw_to_wxyz_arr(pose.orientation_as_array())
            }
        }
        self._write_and_step(write_objects)
        return True

    def reset_multiple_objects_base_poses(self, objects: Dict[Object, Pose]) -> bool:
        objects_data = {obj.name: {"position": pose.position_as_array(),
                                   "quaternion": xyzw_to_wxyz_arr(pose.orientation_as_array())}
                        for obj, pose in objects.items()
                        if self.check_object_exists(obj)}
        if len(objects_data) != len(objects):
            logwarn("object names not found in the simulator.")
            return False
        self._write_and_step(objects_data)
        return True

    def step(self, func: Optional[Callable[[], None]] = None, step_seconds: Optional[float] = None) -> None:
        self.simulator.step()

    def set_link_color(self, link: Link, rgba_color: Color):
        logwarn("Setting link color is not supported in Multiverse.")

    def get_link_color(self, link: Link) -> Color:
        logwarn("Getting link color is not supported in Multiverse.")
        return Color()

    def get_colors_of_object_links(self, obj: Object) -> Dict[str, Color]:
        logwarn("Getting colors of object links is not supported in Multiverse.")
        return {link.name: Color() for link in obj.description.links}

    def set_realtime(self, real_time: bool) -> None:
        pass

    def set_gravity(self, gravity_vector: List[float]) -> None:
        logwarn("Setting gravity is not supported in Multiverse.")

    def disconnect_from_physics_server(self) -> None:
        self.simulator.stop()

    def join_threads(self) -> None:
        pass

    def multiverse_reset_world(self):
        """
        Reset the world using the Multiverse API.
        """
        self.simulator.reset()

    def save_physics_simulator_state(self, state_id: Optional[int] = None, use_same_id: bool = False) -> int:
        logwarn("Saving physics simulator state is not supported in Multiverse.")
        return 0

    def remove_physics_simulator_state(self, state_id: int) -> None:
        logwarn("Removing physics simulator state is not supported in Multiverse.")

    def restore_physics_simulator_state(self, state_id: int) -> None:
        logwarn("Restoring physics simulator state is not supported in Multiverse.")

    def ray_test(self, from_position: List[float], to_position: List[float]) -> int:
        logwarn("Ray test is not supported in Multiverse.")

    def ray_test_batch(self, from_positions: List[List[float]], to_positions: List[List[float]],
                       num_threads: int = 1) -> List[int]:
        logwarn("Ray test batch is not supported in Multiverse.")

    def check_object_exists(self, obj: Object) -> bool:
        """
        Check if the object exists in the Multiverse world.

        :param obj: The object.
        :return: True if the object exists, False otherwise.
        """
        return obj.name in self.simulator.get_all_body_names().result
