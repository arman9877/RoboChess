#! /usr/bin/env python3

###
# This script connects to a Kinova robotic arm using the Kortex API to directly control the actuators' positions,
# calculate forward kinematics (joint angles to Cartesian position), and perform inverse kinematics (Cartesian position to joint angles).
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.Exceptions.KServerException import KServerException

import utilities

# Set up the actuator angles
HOME = [0, 0, 30, 270, 220, 90]
ROBOT_IP = "192.168.1.10"  # Replace with your robot's IP address
USERNAME = "admin"  # Default username
PASSWORD = "admin"  # Default password

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications."""
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def set_actuator_positions_directly(base_client, angles):
    print("Setting actuator angles directly...")

    action = Base_pb2.Action()
    action.name = "Set Actuator Angles"
    action.application_data = ""

    for i, angle in enumerate(angles):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = i
        joint_angle.value = angle

    e = threading.Event()
    notification_handle = base_client.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base_client.ExecuteAction(action)

    print("Waiting for movement to finish...")
    finished = e.wait(TIMEOUT_DURATION)
    base_client.Unsubscribe(notification_handle)

    if finished:
        print("Actuator angles set successfully.")
    else:
        print("Timeout on action notification wait.")
    return finished

def compute_forward_kinematics(base):
    """Compute and display forward kinematics."""
    try:
        input_joint_angles = base.GetMeasuredJointAngles()
        pose = base.ComputeForwardKinematics(input_joint_angles)
        print("\nForward Kinematics Result:")
        print(f"Pose - x: {pose.x}, y: {pose.y}, z: {pose.z}")
        print(f"Orientation - theta_x: {pose.theta_x}, theta_y: {pose.theta_y}, theta_z: {pose.theta_z}\n")
    except KServerException as ex:
        print("Error in computing forward kinematics:", ex)

def compute_inverse_kinematics(base):
    """Compute and display inverse kinematics."""
    try:
        input_joint_angles = base.GetMeasuredJointAngles()
        pose = base.ComputeForwardKinematics(input_joint_angles)
        ik_data = Base_pb2.IKData()

        ik_data.cartesian_pose.x = pose.x
        ik_data.cartesian_pose.y = pose.y
        ik_data.cartesian_pose.z = pose.z
        ik_data.cartesian_pose.theta_x = pose.theta_x
        ik_data.cartesian_pose.theta_y = pose.theta_y
        ik_data.cartesian_pose.theta_z = pose.theta_z

        for joint_angle in input_joint_angles.joint_angles:
            guessed_angle = ik_data.guess.joint_angles.add()
            guessed_angle.value = joint_angle.value - 1

        computed_joint_angles = base.ComputeInverseKinematics(ik_data)

        print("\nInverse Kinematics Result:")
        for i, joint_angle in enumerate(computed_joint_angles.joint_angles):
            print(f"Joint {i}: {joint_angle.value}")
        print()
    except KServerException as ex:
        print("Error in computing inverse kinematics:", ex)

def set_gripper_position(base_client, position):
    """Set the gripper position (0.0 to 1.0)."""
    print(f"Setting gripper to position: {position*100}%")
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position
    base_client.SendGripperCommand(gripper_command)
    time.sleep(2)  # Allow gripper to complete the movement

def main():
    args = type('Args', (object,), {"ip": ROBOT_IP, "username": USERNAME, "password": PASSWORD})()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        try:
            # Create BaseClient instance
            base_client = BaseClient(router)

            set_gripper_position(base_client, 0.6)

            # Set actuator positions
            success = set_actuator_positions_directly(base_client, HOME)
            if success:
                # Compute and display forward kinematics
                compute_forward_kinematics(base_client)

                # Compute and display inverse kinematics
                compute_inverse_kinematics(base_client)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
