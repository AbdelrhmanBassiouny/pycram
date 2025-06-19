from .default_process_modules import DefaultManager, DefaultMoveGripperReal, DefaultMoveGripper
from ..datastructures.enums import Arms, ExecutionType, GripperState
from ..designators.motion_designator import MoveGripperMotion
from ..process_module import ProcessModule, ProcessModuleManager
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from ..datastructures.world import World
from ..robot_description import RobotDescription


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

    def _execute(self, designator: MoveGripperMotion):
        arm_chain = RobotDescription.current_robot_description.get_arm_chain(designator.gripper)
        if designator.motion == GripperState.CLOSE:
            World.robot.set_multiple_joint_positions(arm_chain.get_static_gripper_state(GripperState.PINCH))
        else:
            World.robot.set_multiple_joint_positions(arm_chain.get_static_gripper_state(GripperState.OPEN))
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
