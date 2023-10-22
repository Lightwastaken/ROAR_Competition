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

logging.basicConfig(level=logging.INFO)
matplotlib.use("TkAgg")


class FullView:
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

    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        self.depth_value_array = []
        self.seconds_array = []
        self.sameSecond_array = []

    def render(self, image: roar_py_interface.RoarPyCameraSensorData,
               image2: roar_py_interface.RoarPyCameraSensorDataDepth,
               location: roar_py_interface.RoarPyLocationInWorldSensorData, waypoints) -> Optional[Dict[str, Any]]:
        # print(location)
        # print(waypoints)
        image_pil = image.get_image()
        # plt.rcParams["figure.figsize"] = [20, 20]
        # plt.figure(figsize=(50,50))
        # logging.infO("img_Pil: image = " + str(image_pil))
        image2_np = image2.image_depth
        min, max = 0, 40
        normalized_image = (image2_np) / (max - min)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        image2_pil = Image.fromarray(normalized_image, mode="L")
        # image2_pil = ImageOps.invert(image2_pil)
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)
        depth_value = np.average(image2_np)
        intdp = int(depth_value)
        ticks = pygame.time.get_ticks()
        seconds = int(ticks / 1000)
        display_sec_array = self.seconds_array
        display_depth_array = self.depth_value_array
        self.preSec = seconds
        # if resetSec >= 60:
        #     plt.clf()
        plt.show(block=False)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
                pygame.quit()
                return None

        combined_img_pil = Image.new('RGB', (image_pil.width, image_pil.height), (250, 250, 250))
        combined_img_pil.paste(image_pil, (0, 0))
        # size = canvas.get_width_height()
        # surf = pygame.image.fromstring(raw_data, size , "RGB")

        # combined_img_pil.paste(im,(image_pil.width,image2_pil.height))
        image_surface = pygame.image.fromstring(combined_img_pil.tobytes(), combined_img_pil.size,
                                                combined_img_pil.mode).convert()
        self.screen.fill((0, 0, 0))

        self.screen.blit(image_surface, (0, 0))
        # self.screen.blit(surf, (image_pil.width, image2_pil.height))

        pygame.display.flip()
        self.clock.tick(60)
        return depth_value
