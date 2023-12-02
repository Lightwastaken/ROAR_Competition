import time

import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import transforms3d as tr3d


async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.00, 0.005)
    waypointx = []
    waypointy = []
    print("Map Name", carla_world.map_name)
    waypoints = roar_py_instance.world.maneuverable_waypoints
    spawn_points = roar_py_instance.world.spawn_points
    roar_py_instance.close()
    waypoints = waypoints[::10]
    with plt.ion():

        i = 0
        for waypoint in (waypoints[:] if waypoints is not None else []):
            if i % 2 == 0:
                rep_line = waypoint.line_representation
                rep_line = np.asarray(rep_line)
                print(str(waypoint.location[0]) + " , " + str(waypoint.location[1]))
                waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array([1, 0, 0])
                plt.arrow(
                    waypoint.location[0],
                    waypoint.location[1],
                    waypoint_heading[0] * 1,
                    waypoint_heading[1] * 1,
                    width=0.5,
                    color='r'
                )
                plt.plot(rep_line[:, 0], rep_line[:, 1])
                plt.pause(0.001)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    asyncio.run(main())