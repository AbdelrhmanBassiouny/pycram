from .default_process_modules import DefaultManager, DefaultMoveGripperReal, DefaultMoveGripper
from ..datastructures.enums import Arms, ExecutionType, GripperState
from ..designators.motion_designator import MoveGripperMotion
from ..process_module import ProcessModule, ProcessModuleManager
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from ..datastructures.world import World
from ..robot_description import RobotDescription
from multiverse_client_py import MultiverseClient, MultiverseMetaData

class GripperConnector(MultiverseClient):
    open_state = None
    close_state = None

    def __init__(self, port: str, multiverse_meta_data: MultiverseMetaData) -> None:
        super().__init__(port, multiverse_meta_data)
        self.init()

    def init(self) -> None:
        raise NotImplementedError("init method is not implemented")

    def open_gripper(self):
        send_data = [self.sim_time]
        for object_name in self.response_meta_data["send"].keys():
            joint_name = object_name.replace("actuator", "joint")
            send_data += [self.open_state[joint_name]]
        self.send_data = send_data
        self.send_and_receive_data()

    def close_gripper(self):
        send_data = [self.sim_time]
        for object_name in self.response_meta_data["send"].keys():
            joint_name = object_name.replace("actuator", "joint")
            send_data += [self.close_state[joint_name]]
        self.send_data = send_data
        self.send_and_receive_data()

    def loginfo(self, message: str) -> None:
        print(f"INFO: {message}")

    def logwarn(self, message: str) -> None:
        print(f"WARN: {message}")

    def _run(self) -> None:
        self.loginfo("Start running the client.")
        self._connect_and_start()

    def send_and_receive_meta_data(self) -> None:
        self.loginfo("Sending request meta data: " + str(self.request_meta_data))
        self._communicate(True)
        self.loginfo("Received response meta data: " + str(self.response_meta_data))

    def send_and_receive_data(self) -> None:
        self._communicate(False)

class LeftHandCommand(GripperConnector):
    open_state = RobotDescription.current_robot_description.get_arm_chain(Arms.LEFT).get_static_gripper_state(GripperState.OPEN)
    close_state = RobotDescription.current_robot_description.get_arm_chain(Arms.LEFT).get_static_gripper_state(GripperState.PINCH)

    def init(self) -> None:
        self.run()
        self.request_meta_data["send"] = {}
        for joint_name in self.close_state.keys():
            actuator_name = joint_name.replace("joint", "actuator")
            self.request_meta_data["send"][actuator_name] = [
                "cmd_joint_rvalue",
            ]
        self.send_and_receive_meta_data()

class RightHandCommand(GripperConnector):
    open_state = RobotDescription.current_robot_description.get_arm_chain(Arms.RIGHT).get_static_gripper_state(GripperState.OPEN)
    close_state = RobotDescription.current_robot_description.get_arm_chain(Arms.RIGHT).get_static_gripper_state(GripperState.PINCH)

    def init(self) -> None:
        self.run()
        self.request_meta_data["send"] = {}
        for joint_name in self.close_state.keys():
            actuator_name = joint_name.replace("joint", "actuator")
            self.request_meta_data["send"][actuator_name] = [
                "cmd_joint_rvalue",
            ]
        self.send_and_receive_meta_data()

class ICubManager(DefaultManager):
    _instance: ProcessModuleManager = None
    """
    Singelton instance of this Process Module Manager
    """
    def __init__(self):
        super().__init__("iCub")

    def move_gripper(self):
        if ProcessModuleManager.execution_type == ExecutionType.SIMULATED:
            return DefaultMoveGripper(self._move_gripper_lock)
        elif ProcessModuleManager.execution_type == ExecutionType.REAL:
            return MoveGripperReal(self._move_gripper_lock)


class MoveGripperReal(ProcessModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.right_hand_service = rospy.ServiceProxy("/grasp_with_right_hand", SetBool)
        self.left_hand_service = rospy.ServiceProxy("/grasp_with_left_hand", SetBool)
        multiverse_meta_data = MultiverseMetaData(
            world_name="world",
            simulation_name="left_hand_controller",
            length_unit="m",
            angle_unit="rad",
            mass_unit="kg",
            time_unit="s",
            handedness="rhs",
        )
        self.left_hand_controller = LeftHandCommand(port="3242", multiverse_meta_data=multiverse_meta_data)
        multiverse_meta_data = MultiverseMetaData(
            world_name="world",
            simulation_name="right_hand_controller",
            length_unit="m",
            angle_unit="rad",
            mass_unit="kg",
            time_unit="s",
            handedness="rhs",
        )
        self.right_hand_controller = RightHandCommand(port="3243", multiverse_meta_data=multiverse_meta_data)

    def _execute(self, designator: MoveGripperMotion):
        if designator.motion == GripperState.CLOSE:
            if designator.gripper == Arms.RIGHT:
                self.right_hand_controller.close_gripper()
            else:
                self.left_hand_controller.close_gripper()
        else:
            if designator.gripper == Arms.RIGHT:
                self.right_hand_controller.open_gripper()
            else:
                self.left_hand_controller.open_gripper()
        if designator.gripper == Arms.RIGHT:
            service_name = "/grasp_with_right_hand"
            service = self.right_hand_service
        else:
            service_name = "/grasp_with_left_hand"
            service = self.left_hand_service
        # Wait for the service to be available
        rospy.wait_for_service(service_name)

        try:

            # Create a request object
            request = SetBoolRequest()
            request.data = designator.motion == GripperState.CLOSE  # Set the boolean value to true as in your example

            # Call the service
            response = service(request)

            # Print the response
            rospy.loginfo("Service call successful: %s", response.success)
            rospy.loginfo("Message: %s", response.message)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


ICubManager()
