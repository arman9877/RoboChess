#! /usr/bin/env python3

###
# This script connects to a Kinova robotic arm using the Kortex API to control the actuators' positions directly.
# It sequentially moves the robot through a series of predefined positions, including computing forward 
# and inverse kinematics. The positions are specified as joint angles in degrees, and the script uses 
# high-level API commands to execute each movement.
#
# The robot positions include an initial position, several intermediate waypoints, and a final home position.
# It waits for each movement to complete before proceeding to the next position.
#
# Note: Ensure the Kinova robotic arm is properly set up and powered on before running this script.
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.Exceptions.KServerException import KServerException

import utilities

# Define multiple positions
POSITIONS = [
    [69.57, 326.00, 125.41, 278.93, 340.20, 148],
    [63.53, 329.33, 135.40, 283.50, 346.48, 139.15],
    [60.51, 330.44, 149.62, 270.22, 359.08, 148.81],
    [42.07, 330.88, 150.11, 294.26, 359.22, 110.74],
    [15.71, 331.23, 150.02, 291.81, 358.88, 83.88],
    [353.93, 330.62, 150, 272.72, 359.97, 81.16],
    [334.32, 330.17, 144.11, 264.47, 354.48, 70],
    [318.90, 326.83, 129.76, 270, 342.61, 48.82],
    [323.57, 324.01, 119.29, 278.60, 335.48, 41.47],
    [330.03, 320.27, 109.11, 276.72, 329.78, 54.01],
    [334.66, 315.57, 97.61, 275.47, 323.28, 60.08],
    [337.70, 309.01, 82.68, 274.52, 315.07, 64.28],
    [340.51, 300.98, 64.58, 273.82, 305.23, 68.13],
    [342.54, 290.51, 41.68, 273.33, 292.20, 71.12],
    [344.06, 272.04, 2.17, 272.52, 275.34, 73.76],
    [0, 0, 30, 270, 220, 90]
]

ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"
TIMEOUT_DURATION = 20

def check_for_end_or_abort(e):
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def set_actuator_positions_directly(base_client, angles):
    print(f"Setting actuator angles to: {angles}")

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

def main():
    args = type('Args', (object,), {"ip": ROBOT_IP, "username": USERNAME, "password": PASSWORD})()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        try:
            base_client = BaseClient(router)

            for position in POSITIONS:
                success = set_actuator_positions_directly(base_client, position)
                if not success:
                    print(f"Failed to move to position: {position}")
                    break
                time.sleep(1)  # Add a small delay between moves

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
