import numpy as np
import pygame
import roar_py_carla
import roar_py_interface
from PIL import Image, ImageOps, ImageDraw
import roar_py_interface
from typing import Optional, Dict, Any
import logging
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as mplstyle
import time
import transforms3d as tr3d
import io
import roar_py_interface
import csv
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg
logging.basicConfig(level=logging.INFO)
matplotlib.use("TkAgg")

class GraphWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        central_widget = QWidget()
        central_widget.setFixedHeight(700)
        central_widget.setFixedWidth(500)
        self.setCentralWidget(central_widget)
        self.move(0, 0)
        layout = QtWidgets.QVBoxLayout(central_widget)
        speed_layout = QtWidgets.QHBoxLayout()
        angle_layout = QtWidgets.QHBoxLayout()
        self.angle_graph = pg.PlotWidget()
        self.speed_graph = pg.PlotWidget()

        angle_layout.addWidget(self.angle_graph)
        speed_layout.addWidget(self.speed_graph)

        layout.addLayout(speed_layout)
        layout.addLayout(angle_layout)
        self.current_angle = []
        self.target_angle = []
        self.time_data = []
        self.depth_data = []
        self.time_data2 = []
        self.speed_data = []

        self.angle_graph.setBackground("w")
        self.angle_graph.setLabel("left", "Angle")
        self.angle_graph.setLabel("bottom", "Time (s)")
        self.angle_graph.setYRange(0, 40)
        pen = pg.mkPen(color=(255, 0, 0), width=3)

        self.speed_graph.setBackground("w")
        self.speed_graph.setLabel("left", "Speed")
        self.speed_graph.setLabel("bottom", "Time (s)")
        # self.speed_graph.setYRange(0, 40)
        pen2 = pg.mkPen(color=(0, 0, 255), width=2)

        self.angle_line = self.depth_graph.plot(self.time_data, self.depth_data, pen=pen)
        self.speed_line = self.speed_graph.plot(self.time_data2, self.speed_data, pen=pen2)

    def add_data_depth(self,x,y):
        self.time_data = x[:-1]
        self.depth_data = y[:-1]
        self.depth_line.setData(self.time_data, self.depth_data)

    def add_data_speed(self, x, y):
        self.time_data2 = x[:-1]
        self.speed_data = y[:-1]
        self.speed_line.setData(self.time_data2, self.speed_data)

class PyGameViewer2:
    def __init__(
            self
    ):
        self.ax = None
        self.sameSecond_array = None
        self.preSec = None
        self.seconds_array = None
        self.lines = None
        self.depth_value_array = None
        self.figure = None
        self.screen = None
        self.clock = None
        self.currentSpeedArray = None
        self.TargetSpeedArray = None
        self.sameSecondCurrentSpeedArray = None
        self.app = QtWidgets.QApplication([])
        self.main = GraphWindow()
        self.main.show()
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        # self.figure, self.ax = plt.subplots(nrows=1, ncols=1)
        # self.lines = self.ax.plot(1, 50, 'bo', markersize=3)[0]
        self.currentSpeedArray = []
        self.TargetSpeedArray = []
        self.sameSecondCurrentSpeedArray = []
        self.depth_value_array = []
        self.seconds_array = []
        self.sameSecond_array = []
        self.preSec = 0

    def render(self, image: roar_py_interface.RoarPyCameraSensorData,
               image2,
               occupancy_map: Image,
               location: roar_py_interface.RoarPyLocationInWorldSensorData, waypoints, current_speed)-> Optional[Dict[str, Any]]:
        # print(location)
        # print(waypoints)
        image_pil = image.get_image()
        # plt.rcParams["figure.figsize"] = [20, 20]
        # plt.figure(figsize=(50,50))
        # logging.infO("img_Pil: image = " + str(image_pil))
        image2_np = image2.image_depth
        # logging.infO("img_Pil:Depth = " + str(image_pil))
        image2_np = image2_np[100:150, len(image2_np) - 50: len(image2_np) + 50]
        # image2_np = np.log(image2_np)
        image2_np = np.clip(image2_np, 0, 40)
        min, max = np.min(image2_np), np.max(image2_np)
        min, max = 0, 40
        normalized_image = (image2_np) / (max - min)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        image2_pil = Image.fromarray(normalized_image, mode="L")
        occupancy_map_rgb = occupancy_map.convert("RGB") if occupancy_map is not None else None
        # image2_pil = ImageOps.invert(image2_pil)
        # jogn waas here
        if self.screen is None:
            self.init_pygame(image_pil.width + image2_pil.width + occupancy_map.width, image_pil.height)
        # mplstyle.use('fast')
        depth_value = np.average(image2_np)
        intdp = int(depth_value)
        depth_value_text = ImageDraw.Draw(image2_pil)
        depth_value_text.text((2, 2), str(depth_value), fill=0)
        ticks = pygame.time.get_ticks()
        seconds = int(ticks / 1000)
        resetSec = int((ticks / 1000) % 60)
        logging.info("sec: " + str(seconds) + ", depth:" + str(intdp))
        if seconds != self.preSec:
            self.seconds_array.append(seconds)
            self.depth_value_array.append(depth_value)
            self.currentSpeedArray.append(current_speed)
            self.sameSecond_array = []
        if seconds > 0 and seconds == self.seconds_array[-1]:
            self.sameSecond_array.append(depth_value)
            self.sameSecondCurrentSpeedArray.append(current_speed)
            # print('dp appened')
        if seconds > 0 and len(self.sameSecond_array) > 0:
            self.sameSecond_array.sort()
            self.sameSecondCurrentSpeedArray.sort()
            print("same sec: " + str(self.sameSecond_array))
            self.depth_value_array.append(self.sameSecond_array[0])
            self.depth_value_array.append(self.sameSecond_array[-1])
            self.currentSpeedArray.append(self.sameSecondCurrentSpeedArray[0])
            self.currentSpeedArray.append(self.sameSecondCurrentSpeedArray[-1])
            self.seconds_array.append(seconds)
            self.seconds_array.append(seconds)
        # the code above lowk does not work
        # it basically tries to add the least and greatest value within a second to our array(which will be graphed) to help w/ performance
        # debug code
        # print("seconds arr:" + str(self.depth_value_array))
        # print("value arr" + str(self.seconds_array))
        display_sec_array = self.seconds_array
        display_depth_array = self.depth_value_array
        self.preSec = seconds
        # if graphnum == 1:
        #     self.lines.set_xdata(display_sec_array)
        #     self.lines.set_ydata(display_depth_array)
        #     plt.plot(display_sec_array, display_depth_array)
        # if graphnum == 2:
        #     self.lines.set_xdata(display_sec_array)
        #     self.lines.set_ydata(self.currentSpeedArray)
        #     plt.plot(display_sec_array, self.currentSpeedArray)
        self.main.add_data_depth(display_sec_array, display_depth_array)
        self.main.add_data_speed(display_sec_array, self.currentSpeedArray)

        #plt.show(block=False)

        # plt.show(block=False)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.main.close()
                f = open('testlogs.txt', 'a+')
                try:
                    contents = f.readlines()
                    a = contents.count("~")
                except:
                    f.write("attempt: " + '0' + '~')
                f.write("attempt: " + str(a) + '~')
                f.write('\n')
                f.write(str(self.seconds_array))
                f.write('\n')
                f.write(str(self.depth_value_array))
                f.write('\n')
                f.write('-----------------------------------------')
                f.write('\n')
                # plt.close('all')
                pygame.quit()
                return None

        combined_img_pil = Image.new('RGB', (image_pil.width + image2_pil.width + occupancy_map.width, image_pil.height), (250, 250, 250))
        combined_img_pil.paste(image_pil, (0, 0))
        # combined_img_pil.paste(image2_pil, (image_pil.width, 0))
        # combined_img_pil.paste(occupancy_map, (image_pil.width + image2_pil.width, 0))
        # size = canvas.get_width_height()
        # surf = pygame.image.fromstring(raw_data, size , "RGB")

        # combined_img_pil.paste(im,(image_pil.width,image2_pil.height))
        image_surface = pygame.image.fromstring(combined_img_pil.tobytes(), combined_img_pil.size,
                                                combined_img_pil.mode).convert()

        # if occupancy_map_rgb is not None:
        #     occupancy_map_surface = pygame.image.fromstring(occupancy_map_rgb.tobytes(), occupancy_map_rgb.size, occupancy_map_rgb.mode).convert()
        self.screen.fill((0, 0, 0))

        self.screen.blit(image_surface, (0, 0))
        # if occupancy_map_rgb is not None:
        #     self.screen.blit(occupancy_map_surface, (image_pil.width, 0))
        # self.screen.blit(surf, (image_pil.width, image2_pil.height))

        pygame.display.flip()
        self.clock.tick(60)
        return depth_value