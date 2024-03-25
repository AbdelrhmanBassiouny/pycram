#!/usr/bin/env python3

import time
import unittest

from typing_extensions import Optional

from pycram.worlds.multiverse import Multiverse
from pycram.datastructures.enums import ObjectType
from pycram.datastructures.pose import Pose
from pycram.object_descriptors.urdf import ObjectDescription
from pycram.world_concepts.world_object import Object

multiverse_installed = True
try:
    from pycram.world_concepts.multiverse_socket import MultiverseMetaData, SocketAddress
except ImportError:
    multiverse_installed = False


@unittest.skipIf(not multiverse_installed, "Multiverse is not installed.")
# @unittest.skip("Needs Multiverse server and simulation to be running")
class MultiversePyCRAMTestCase(unittest.TestCase):
    multiverse: Multiverse
    table: Optional[Object] = None
    robot: Optional[Object] = None

    @classmethod
    def setUpClass(cls):
        cls.multiverse = Multiverse(simulation="crane_simulation",
                                    client_addr=SocketAddress(port="5481"),
                                    is_prospection=True)

        # cls.table = Object("wooden_log_1", ObjectType.GENERIC_OBJECT, "WoodenLog.stl", ObjectDescription,
        #                    pose=Pose([-3.17, 4, 1], [0, 0, 0, 1]))
        cls.robot = Object("ur5e", ObjectType.ROBOT, "ur5e_without_gripper.urdf", ObjectDescription,
                           pose=Pose())

    @classmethod
    def tearDownClass(cls):
        cls.multiverse.disconnect_from_physics_server()

    @classmethod
    def tearDown(cls):
        if cls.table is not None:
            cls.table.set_position([-3.17, 4, 1])

    def test_set_position(self):
        table_position = self.table.get_position_as_list()
        self.assertEqual(table_position, [-3.17, 4, 1])
        table_position[0] += 1
        self.table.set_position(table_position)
        table_position = self.table.get_position_as_list()
        self.assertAlmostEqual(table_position, [-2.17, 4, 1])
        time.sleep(5)

    def test_update_position(self):
        self.table.update_pose()
        table_position = self.table.get_position_as_list()
        for i, v in enumerate([-3.17, 4, 1]):
            self.assertAlmostEqual(table_position[i], v)

    def test_get_joint_position(self):
        joint_position = self.robot.get_joint_position("shoulder_pan_joint")
        self.assertAlmostEqual(joint_position, 0.0)

    def test_set_joint_position(self):
        joint_position = self.robot.get_joint_position("shoulder_pan_joint")
        self.robot.set_joint_position("shoulder_pan_joint", joint_position - 1.0)
        self.robot.joints["shoulder_pan_joint"]._update_position()
        joint_position = self.robot.get_joint_position("shoulder_pan_joint")
        self.assertAlmostEqual(joint_position, -1.0)

    def test_set_robot_position(self):
        self.robot.set_position([0, 0, 1])
        self.assertEqual(self.robot.get_position_as_list(), [0, 0, 1])

