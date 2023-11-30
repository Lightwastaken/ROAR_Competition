"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
import time
import json
import logging



class ZoneController:
    def __init__(self):
        # Define zones and corresponding control parameters
        self.zone_mapping = {
            "zone1": {"throttle": 0.9, "steer": 0.05},
            "zone2": {"throttle": 0.5, "steer": 0.2},
            "zone3": {"throttle": 0.3, "steer": -0.2},
        }
        ## this is not being used currently and can be changed if needed

    def get_current_zone(self, car_location):
        # Implement logic to determine the current zone based on car's location
        if car_location[0] < -330 or (-200.0 <= car_location[0] < 300 and 700.0 <= car_location[1] < 1000) or (
                -130 <= car_location[0] < -80 and -1000.0 <= car_location[1] < 0) or (
                50 <= car_location[0] < 725 and 100.0 <= car_location[1] < 650) or (
                400 < car_location[0] < 650 and 980 < car_location[1] < 1080) or (
                745 < car_location[0] < 765 and 850 < car_location[1] < 970) or (
                700 < car_location[0] < 900 and 830 < car_location[0] < 1000):
            return 1
        elif (car_location[0] < -125 and 450 < car_location[1] < 820) or (
                700 < car_location[0] and 710 < car_location[1]) or (
                car_location[0] < -130 and car_location[1] < -650) or (
                -350 < car_location[0] < -230 and 390 < car_location[1] < 850):
            return 2
        elif (640 < car_location[0] and 1000 < car_location[1]) or (730 < car_location[0] and 720 < car_location[1] < 830):
            return 4
        else:
            return 3


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )

    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution_MAIN:
    def __init__(
            self,
            maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
            vehicle: roar_py_interface.RoarPyActor,
            camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
            location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
            velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
            rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
            occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
            collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.stopThrottle = None
        self.handbrake = None
        self.data = None
        self.K_val_thresholds = None
        self.prev_key = None
        self.steer_integral_error_prior = None
        self.steer_error_prior = None
        self.error_prior = None
        self.integral_prior = None
        self.start_time = None
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.
        self.stopThrottle = False
        self.handbrake = 0
        self.start_time = time.time()
        self.ZoneControl = ZoneController()
        self.integral_prior = 0
        self.steer_error_prior = 0
        self.error_prior = 0
        self.steer_integral_error_prior = 0
        self.prev_key = 14
        self.K_val_thresholds = []
        with open('PIDconfig.json') as json_file:
            self.data = json.load(json_file)
        controller_values = self.data["Throttle_Controller"]
        for key in controller_values:
            self.K_val_thresholds.append(int(key))

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

    async def step(
            self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + 10) % len(self.maneuverable_waypoints)]
        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        print("rotation" + str(vehicle_rotation[2]))
        print(heading_to_waypoint)
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
        curr_Speed = (int(vehicle_velocity_norm))
        # appending our current speed in an array with all the speed boundaries in PID config
        self.K_val_thresholds.append(curr_Speed)
        # sort the array from least to greatest
        self.K_val_thresholds.sort()
        try:
            # we are getting the K parameter's that is greater than our current speed by one
            K_Values_Determinant = self.K_val_thresholds[self.K_val_thresholds.index(curr_Speed) + 1]
        except:
            # array overflow fix
            K_Values_Determinant = self.K_val_thresholds[self.K_val_thresholds.index(curr_Speed)]
        # setting K vals according to our determinant
        K_Values_Determinant = str(K_Values_Determinant)
        Skp = self.data["Steer_Controller"][K_Values_Determinant]["Kp"]
        Ski = self.data["Steer_Controller"][K_Values_Determinant]["Ki"]
        Skd = self.data["Steer_Controller"][K_Values_Determinant]["Kd"]
        Kp = self.data["Throttle_Controller"][K_Values_Determinant]["Kp"]
        Ki = self.data["Throttle_Controller"][K_Values_Determinant]["Ki"]
        Kd = self.data["Throttle_Controller"][K_Values_Determinant]["Kd"]
        print(K_Values_Determinant)
        # we remove our current speed val, so we can reuse this algo
        self.K_val_thresholds.remove(curr_Speed)
        # Proportional controller to steer the vehicle towards the target waypoint, normal implementation
        steer_error = delta_heading / np.pi
        iteration_time = time.time() - self.start_time
        steer_integral = self.steer_integral_error_prior + steer_error
        steer_derivative = (steer_error - self.steer_error_prior)
        # square rooting the velocity makes it so that higher speed lower steer applied I think i chatgpted this
        Sensitivity = np.sqrt(vehicle_velocity_norm)

        steer_control = (
                Skp * delta_heading / np.pi + (Ski * steer_integral) + (Skd * steer_derivative)
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        print("steer control" + str(steer_control))
        self.steer_integral_error_prior = steer_integral
        self.steer_error_prior = steer_error
        current_speed = vehicle_velocity_norm
        print(current_speed)
        # normal implementation of throttle algo
        target_speed = 45
        if self.ZoneControl.get_current_zone(vehicle_location) == 3:
            target_speed = 20
            if current_speed > target_speed:
                self.stopThrottle = True
                print("BREAK BREAK")
                self.handbrake = 1
            else:
                self.stopThrottle = False
                self.handbrake = 0
            print("ZONE DETECTED 3")
            print("ZONE DETECTED 3")
        elif self.ZoneControl.get_current_zone(vehicle_location) == 2:
            target_speed = 45
            print("ZONE DETECTED 2")
            print("ZONE DETECTED 2")
        elif self.ZoneControl.get_current_zone(vehicle_location) == 1:
            target_speed = 50
            print("ZONE DETECTED 1")
            print("ZONE DETECTED 1")
        current_speed = vehicle_velocity_norm
        error = target_speed - current_speed
        derivative = (error - self.error_prior) / iteration_time
        integral = self.integral_prior + error * iteration_time
        i_value = Ki * integral
        if error != self.error_prior:
            integral = 0

        throttle_control = Kp * error + Ki * integral + Kd * derivative
        if self.stopThrottle == True:
            throttle_control = 0

        # apply anti-windup???
        gear = max(1, (current_speed // 10))
        if throttle_control == -1:
            gear = -1

        print("speed: " + str(vehicle_velocity_norm))
        self.error_prior = error
        self.integral_prior = integral



        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": self.handbrake,
            "reverse": 0,
            "target_gear": gear
        }
        await self.vehicle.apply_action(control)
        return control
