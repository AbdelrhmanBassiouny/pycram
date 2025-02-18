import os

from pycram.datastructures.dataclasses import Color
from pycram.datastructures.enums import Arms, WorldMode, Grasp
from pycram.datastructures.pose import Pose
from pycram.designators.action_designator import ParkArmsAction, PickUpAction
from pycram.designators.object_designator import BelieveObject
from pycram.helper import find_multiverse_resources_path
from pycram.object_descriptors.generic import ObjectDescription as GenericObjectDescription
from pycram.object_descriptors.urdf import ObjectDescription
from pycram.process_module import simulated_robot
from pycram.world_concepts.world_object import Object
from pycram.worlds.multiverse2 import Multiverse
from pycrap.ontologies import Robot, PhysicalObject

resources_path = find_multiverse_resources_path()
example_scene_path = os.path.join(resources_path,
                                  "robots/mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
multiverse = Multiverse(scene_file_path=example_scene_path,
                        mode=WorldMode.GUI, prospection_mode=WorldMode.DIRECT)
extension = ObjectDescription.get_file_extension()

multiverse.step()
robot = Object('panda', Robot, f'panda{extension}')

obj_desc = GenericObjectDescription('box', [0, 0, 0], [0.02, 0.02, 0.02],
                                    color=Color(0, 1, 0, 1))
box = Object("box", PhysicalObject, None, description=obj_desc,
             pose=Pose([0.7, 0, 0.03]))
multiverse.step()

robot_desig = BelieveObject(names=[robot.name])

with simulated_robot:
    print([j.position for j in robot.joints.values()])
    # Transport the milk
    ParkArmsAction([Arms.BOTH]).resolve().perform()
    multiverse.step()

    print([j.position for j in robot.joints.values()])
    box_desig = BelieveObject(names=[box.name])
    PickUpAction(box_desig, [Arms.RIGHT], [Grasp.TOP, Grasp.FRONT, Grasp.BACK]).resolve().perform()

multiverse.exit()
