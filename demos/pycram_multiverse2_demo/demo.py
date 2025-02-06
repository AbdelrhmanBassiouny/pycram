import os

import pycrap
from pycram.datastructures.dataclasses import Color
from pycram.datastructures.enums import Arms, WorldMode
from pycram.datastructures.pose import Pose
from pycram.designators.action_designator import ParkArmsAction, MoveTorsoAction, TransportAction, NavigateAction, \
    LookAtAction
from pycram.designators.object_designator import BelieveObject
from pycram.helper import find_multiverse_resources_path
from pycram.object_descriptors.urdf import ObjectDescription
from pycram.object_descriptors.generic import ObjectDescription as GenericObjectDescription
from pycram.process_module import simulated_robot
from pycram.world_concepts.world_object import Object
from pycram.worlds.multiverse2 import Multiverse

resources_path = find_multiverse_resources_path()
example_scene_path = os.path.join(resources_path,
                                  "robots/mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
multiverse = Multiverse(scene_file_path=example_scene_path,
                        mode=WorldMode.GUI, prospection_mode=WorldMode.GUI)
extension = ObjectDescription.get_file_extension()
robot = Object('panda', pycrap.Robot, f'panda{extension}')

obj_desc = GenericObjectDescription('box', [0, 0, 0], [0.02, 0.02, 0.02],
                                    color=Color(0, 1, 0, 1))
box = Object("box", pycrap.PhysicalObject, None, description=obj_desc)

robot_desig = BelieveObject(names=[robot.name])

with simulated_robot:
    # Transport the milk
    ParkArmsAction([Arms.BOTH]).resolve().perform()

    NavigateAction(target_locations=[Pose([1.7, 2, 0])]).resolve().perform()

    box_desig = BelieveObject(names=[box.name])
    TransportAction(box_desig, [Pose([2.4, 3, 1.02])], [Arms.RIGHT]).resolve().perform()

    exit(0)
    # Find and navigate to the drawer containing the spoon
    handle_desig = ObjectPart(names=["cabinet10_drawer1_handle"], part_of=apartment_desig.resolve())
    drawer_open_loc = AccessingLocation(handle_desig=handle_desig.resolve(),
                                        robot_desig=robot_desig.resolve()).resolve()

    NavigateAction([drawer_open_loc.pose]).resolve().perform()

    OpenAction(object_designator_description=handle_desig,
               arms=[drawer_open_loc.arms[0]]).resolve().perform()

    arm_ee = RobotDescription.current_robot_description.get_arm_chain(drawer_open_loc.arms[0]).get_tool_frame()
    closing_arm_pose = robot.get_link_pose(arm_ee)

    # Detect and pickup the spoon
    spoon.detach(apartment)
    LookAtAction([apartment.get_link_pose("cabinet10_drawer1_handle")]).resolve().perform()

    spoon_desig = DetectAction(DetectionTechnique.TYPES,
                               object_designator_description=BelieveObject(types=[pycrap.Spoon])).resolve().perform()[0]

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    pickup_arm = Arms.LEFT if drawer_open_loc.arms[0] == Arms.RIGHT else Arms.RIGHT
    pick_up_loc = CostmapLocation(target=spoon_desig.pose, reachable_for=robot_desig.resolve(),
                                  reachable_arm=pickup_arm, grasps=[Grasp.TOP]).resolve()

    NavigateAction([pick_up_loc.pose]).resolve().perform()
    MoveTCPMotion(closing_arm_pose, drawer_open_loc.arms[0]).perform()

    PickUpAction(spoon_desig, [pickup_arm], [Grasp.TOP]).resolve().perform()

    ParkArmsAction([Arms.LEFT if pickup_arm == Arms.LEFT else Arms.RIGHT]).resolve().perform()

    NavigateAction([drawer_open_loc.pose]).resolve().perform()

    CloseAction(object_designator_description=handle_desig, arms=[drawer_open_loc.arms[0]]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

    MoveTorsoAction([0.15]).resolve().perform()

    # Find a pose to place the spoon, move and then place it
    spoon_target_pose = Pose([2.35, 2.6, 0.95], [0, 0, 0, 1])
    placing_loc = CostmapLocation(target=spoon_target_pose,
                                  reachable_for=robot_desig.resolve()).resolve()

    NavigateAction([placing_loc.pose]).resolve().perform()

    PlaceAction(spoon_desig, [spoon_target_pose], [pickup_arm]).resolve().perform()

    ParkArmsAction([Arms.BOTH]).resolve().perform()

world.exit()


def debug_place_spoon():
    robot.set_pose(Pose([1.66, 2.56, 0.01], [0.0, 0.0, -0.04044101807153309, 0.9991819274072855]))
    spoon.set_pose(Pose([1.9601757566599975, 2.06718019258626, 1.0427727691273496],
                        [0.11157527804553227, -0.7076243776942466, 0.12773644958149588, 0.685931554070963]))
    robot.attach(spoon, 'r_gripper_tool_frame')
    pickup_arm = Arms.RIGHT
    spoon_desig = BelieveObject(names=[spoon.name])
    return pickup_arm, spoon_desig
