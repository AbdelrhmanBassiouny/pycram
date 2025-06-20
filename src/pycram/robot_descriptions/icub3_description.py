from ..datastructures.enums import Arms, GripperState, GripperType, TorsoState, StaticJointState
from ..robot_description import RobotDescription, KinematicChainDescription, EndEffectorDescription, \
    RobotDescriptionManager, CameraDescription
from ..ros import get_ros_package_path

filename = get_ros_package_path('pycram') + '/resources/robots/' + "iCub" + '.urdf'

icub_description = RobotDescription("iCub", "base_footprint", "torso_1", "torso_pitch",
                                    filename)

################################## Left Arm ##################################
left_arm = KinematicChainDescription("left", "root_link", "l_hand",
                                     icub_description.urdf_object, arm_type=Arms.LEFT)

# left_arm.add_static_joint_states(StaticJointState.Park, {"torso_roll": 0,
#                                                          "torso_pitch": 0,
#                                                          "torso_yaw": 0,
#                                                          "l_shoulder_pitch": 0,
#                                                          "l_shoulder_roll": 0,
#                                                          "l_shoulder_yaw": 0,
#                                                          "l_elbow": 0,
#                                                          "l_wrist_prosup": 0,
#                                                          "l_wrist_pitch": 0,
#                                                          "l_wrist_yaw": 0})

# left_arm.add_static_joint_states(StaticJointState.Park, {"torso_roll": 0,
#                                                          "torso_pitch": 0.5,
#                                                          "torso_yaw": 0,
#                                                          'l_elbow': 1.309, 'l_shoulder_pitch': -2.0,
#                                                          'l_shoulder_roll': 0.42, 'l_shoulder_yaw': -0.21,
#                                                          'l_wrist_pitch': 0,
#                                                          'l_wrist_prosup': 0,
#                                                          'l_wrist_yaw': 0})
left_arm.add_static_joint_states(StaticJointState.Park,
                                 {'l_elbow': 0.6490029564319394, 'l_shoulder_pitch': -0.8930239720790607,
                                  'l_shoulder_roll': 1.0800233409269198, 'l_shoulder_yaw': 0.22298555297627984,
                                  'l_wrist_pitch': -1.683321743882272e-05, 'l_wrist_prosup': -4.04517285760325e-06,
                                  'l_wrist_yaw': -1.0931676576013595e-05, 'torso_pitch': 0.49990245568463487,
                                  'torso_roll': -6.709279361168048e-05, 'torso_yaw': -3.0876466462262046e-05})

icub_description.add_kinematic_chain_description(left_arm)

################################## Left Gripper ##################################
left_gripper = EndEffectorDescription("left_gripper", "l_hand", "l_gripper_tool_frame",
                                      icub_description.urdf_object)
left_gripper.add_static_joint_states(GripperState.OPEN, {"l_hand_thumb_0_joint": 0.0,
                                                         "l_hand_thumb_1_joint": 0.0,
                                                         "l_hand_thumb_2_joint": 0.0,
                                                         "l_hand_thumb_3_joint": 0.0,
                                                         "l_hand_index_0_joint": 0.0,
                                                         "l_hand_index_1_joint": 0.0,
                                                         "l_hand_index_2_joint": 0.0,
                                                         "l_hand_index_3_joint": 0.0,
                                                         "l_hand_middle_0_joint": 0.0,
                                                         "l_hand_middle_1_joint": 0.0,
                                                         "l_hand_middle_2_joint": 0.0,
                                                         "l_hand_middle_3_joint": 0.0,
                                                         "l_hand_ring_0_joint": 0.0,
                                                         "l_hand_ring_1_joint": 0.0,
                                                         "l_hand_ring_2_joint": 0.0,
                                                         "l_hand_ring_3_joint": 0.0,
                                                         "l_hand_little_0_joint": 0.0,
                                                         "l_hand_little_1_joint": 0.0,
                                                         "l_hand_little_2_joint": 0.0,
                                                         "l_hand_little_3_joint": 0.0})
left_gripper.add_static_joint_states(GripperState.CLOSE, {"l_hand_thumb_0_joint": 1.5707963267948966,
                                                          "l_hand_thumb_1_joint": 1.5707963267948966,
                                                          "l_hand_thumb_2_joint": 1.5707963267948966,
                                                          "l_hand_thumb_3_joint": 1.5707963267948966,
                                                          "l_hand_index_0_joint": -0.3490658503988659,
                                                          "l_hand_index_1_joint": 1.5707963267948966,
                                                          "l_hand_index_2_joint": 1.5707963267948966,
                                                          "l_hand_index_3_joint": 1.5707963267948966,
                                                          "l_hand_middle_0_joint": 0.3490658503988659,
                                                          "l_hand_middle_1_joint": 1.5707963267948966,
                                                          "l_hand_middle_2_joint": 1.5707963267948966,
                                                          "l_hand_middle_3_joint": 1.5707963267948966,
                                                          "l_hand_ring_0_joint": 0.3490658503988659,
                                                          "l_hand_ring_1_joint": 1.5707963267948966,
                                                          "l_hand_ring_2_joint": 1.5707963267948966,
                                                          "l_hand_ring_3_joint": 1.5707963267948966,
                                                          "l_hand_little_0_joint": 0.3490658503988659,
                                                          "l_hand_little_1_joint": 1.5707963267948966,
                                                          "l_hand_little_2_joint": 1.5707963267948966,
                                                          "l_hand_little_3_joint": 1.5707963267948966})
left_gripper.add_static_joint_states(GripperState.PINCH, {"l_hand_thumb_0_joint": 1.26,
                                                           "l_hand_thumb_1_joint": 0.267*0.9,
                                                           "l_hand_thumb_2_joint": 0.353*0.9,
                                                           "l_hand_thumb_3_joint": 0.676*0.9,
                                                           "l_hand_index_0_joint": 0,
                                                           "l_hand_index_1_joint": 0.856,
                                                           "l_hand_index_2_joint": 0.613,
                                                           "l_hand_index_3_joint": 0.251,
                                                           "l_hand_middle_0_joint": 0,
                                                           "l_hand_middle_1_joint": 0.848,
                                                           "l_hand_middle_2_joint": 0.597,
                                                           "l_hand_middle_3_joint": 0.393,
                                                           "l_hand_ring_0_joint": 0,
                                                           "l_hand_ring_1_joint": 0,
                                                           "l_hand_ring_2_joint": 0,
                                                           "l_hand_ring_3_joint": 0,
                                                           "l_hand_little_0_joint": 0,
                                                           "l_hand_little_1_joint": 0,
                                                           "l_hand_little_2_joint": 0,
                                                           "l_hand_little_3_joint": 0})


left_gripper.end_effector_type = GripperType.FINGER
# left_gripper.opening_distance = 0.548
left_arm.end_effector = left_gripper

################################## Right Arm ##################################
right_arm = KinematicChainDescription("right", "root_link", "r_hand",
                                      icub_description.urdf_object, arm_type=Arms.RIGHT)

# right_arm.add_static_joint_states(StaticJointState.Park, {"torso_roll": 0,
#                                                           "torso_pitch": 0,
#                                                           "torso_yaw": 0,
#                                                           "r_shoulder_pitch": 0,
#                                                           "r_shoulder_roll": 0,
#                                                           "r_shoulder_yaw": 0,
#                                                           "r_elbow": 0,
#                                                           "r_wrist_prosup": 0,
#                                                           "r_wrist_pitch": 0,
#                                                           "r_wrist_yaw": 0})

# right_arm.add_static_joint_states(StaticJointState.Park, {
#     'r_shoulder_pitch': 2.0,
#     'r_shoulder_roll': 0.42,
#     'r_shoulder_yaw': -0.21,
#     'r_wrist_pitch': 0,
#     'r_wrist_prosup': 0,
#     'r_wrist_yaw': 0,
#     'torso_pitch': 0.5,
#     'torso_roll': 0,
#     'torso_yaw': 0})
right_arm.add_static_joint_states(StaticJointState.Park,
                                  {'r_elbow': 0.6490000000000863, 'r_shoulder_pitch': 0.8930000000002406,
                                   'r_shoulder_roll': 1.0799999999996839, 'r_shoulder_yaw': 0.2230000000003703,
                                   'r_wrist_pitch': 1.505646045310829e-13, 'r_wrist_prosup': 4.5600715485667593e-14,
                                   'r_wrist_yaw': 7.080575236241232e-14, 'torso_pitch': 0.49990245568463487,
                                   'torso_roll': -6.709279361108927e-05, 'torso_yaw': -3.0876466459611924e-05})
icub_description.add_kinematic_chain_description(right_arm)

################################## Right Gripper ##################################
right_gripper = EndEffectorDescription("right_gripper", "r_hand", "r_gripper_tool_frame",
                                       icub_description.urdf_object)
right_gripper.add_static_joint_states(GripperState.OPEN, {"r_hand_thumb_0_joint": 0.0,
                                                          "r_hand_thumb_1_joint": 0.0,
                                                          "r_hand_thumb_2_joint": 0.0,
                                                          "r_hand_thumb_3_joint": 0.0,
                                                          "r_hand_index_0_joint": 0.0,
                                                          "r_hand_index_1_joint": 0.0,
                                                          "r_hand_index_2_joint": 0.0,
                                                          "r_hand_index_3_joint": 0.0,
                                                          "r_hand_middle_0_joint": 0.0,
                                                          "r_hand_middle_1_joint": 0.0,
                                                          "r_hand_middle_2_joint": 0.0,
                                                          "r_hand_middle_3_joint": 0.0,
                                                          "r_hand_ring_0_joint": 0.0,
                                                          "r_hand_ring_1_joint": 0.0,
                                                          "r_hand_ring_2_joint": 0.0,
                                                          "r_hand_ring_3_joint": 0.0,
                                                          "r_hand_little_0_joint": 0.0,
                                                          "r_hand_little_1_joint": 0.0,
                                                          "r_hand_little_2_joint": 0.0,
                                                          "r_hand_little_3_joint": 0.0})
right_gripper.add_static_joint_states(GripperState.CLOSE, {"r_hand_thumb_0_joint": 1.5707963267948966,
                                                           "r_hand_thumb_1_joint": 1.5707963267948966,
                                                           "r_hand_thumb_2_joint": 1.5707963267948966,
                                                           "r_hand_thumb_3_joint": 1.5707963267948966,
                                                           "r_hand_index_0_joint": -0.3490658503988659,
                                                           "r_hand_index_1_joint": 1.5707963267948966,
                                                           "r_hand_index_2_joint": 1.5707963267948966,
                                                           "r_hand_index_3_joint": 1.5707963267948966,
                                                           "r_hand_middle_0_joint": 0.3490658503988659,
                                                           "r_hand_middle_1_joint": 1.5707963267948966,
                                                           "r_hand_middle_2_joint": 1.5707963267948966,
                                                           "r_hand_middle_3_joint": 1.5707963267948966,
                                                           "r_hand_ring_0_joint": 0.3490658503988659,
                                                           "r_hand_ring_1_joint": 1.5707963267948966,
                                                           "r_hand_ring_2_joint": 1.5707963267948966,
                                                           "r_hand_ring_3_joint": 1.5707963267948966,
                                                           "r_hand_little_0_joint": 0.3490658503988659,
                                                           "r_hand_little_1_joint": 1.5707963267948966,
                                                           "r_hand_little_2_joint": 1.5707963267948966,
                                                           "r_hand_little_3_joint": 1.5707963267948966})

right_gripper.add_static_joint_states(GripperState.PINCH, {"r_hand_thumb_0_joint": 1.26,
                                                           "r_hand_thumb_1_joint": 0.267,
                                                           "r_hand_thumb_2_joint": 0.353,
                                                           "r_hand_thumb_3_joint": 0.676,
                                                           "r_hand_index_0_joint": 0,
                                                           "r_hand_index_1_joint": 0.856,
                                                           "r_hand_index_2_joint": 0.613,
                                                           "r_hand_index_3_joint": 0.251,
                                                           "r_hand_middle_0_joint": 0,
                                                           "r_hand_middle_1_joint": 0.848,
                                                           "r_hand_middle_2_joint": 0.597,
                                                           "r_hand_middle_3_joint": 0.393,
                                                           "r_hand_ring_0_joint": 0,
                                                           "r_hand_ring_1_joint": 0,
                                                           "r_hand_ring_2_joint": 0,
                                                           "r_hand_ring_3_joint": 0,
                                                           "r_hand_little_0_joint": 0,
                                                           "r_hand_little_1_joint": 0,
                                                           "r_hand_little_2_joint": 0,
                                                           "r_hand_little_3_joint": 0})

right_gripper.end_effector_type = GripperType.FINGER
# right_gripper.opening_distance = 0.548
right_arm.end_effector = right_gripper

################################## Torso ##################################
# redo to use knees instead of torso
torso = KinematicChainDescription("torso", "root_link", "chest",
                                  icub_description.urdf_object)

torso.add_static_joint_states(TorsoState.HIGH, {"torso_roll": 0})

torso.add_static_joint_states(TorsoState.MID, {"torso_roll": 0})

torso.add_static_joint_states(TorsoState.LOW, {"torso_roll": 0})

icub_description.add_kinematic_chain_description(torso)

################################## Camera ##################################
# real camera unknown at the moment of writing (also missing in urdf), so using dummy camera for now
camera = CameraDescription("eye_camera", "head", 1.27,
                           1.85, 0.99483, 0.75049,
                           [1, 0, 0])
icub_description.add_camera_description(camera)

################################## Neck ##################################
icub_description.add_kinematic_chain("neck", "chest", "head")
icub_description.set_neck(yaw_joint="neck_yaw", pitch_joint="neck_pitch", roll_joint="neck_roll")

################################# Grasps ##################################
# left_orientation = [0.5, 0.5, 0.5, 0.5]
left_orientation = [0, 0, 0.7071068, 0.7071068]
left_gripper.update_all_grasp_orientations(left_orientation)

# right_orientation = [0, 0, 1, 0]
right_orientation = [0, 0, 0.7071068, 0.7071068]
right_gripper.update_all_grasp_orientations(right_orientation)

# Add to RobotDescriptionManager
rdm = RobotDescriptionManager()
rdm.register_description(icub_description)
