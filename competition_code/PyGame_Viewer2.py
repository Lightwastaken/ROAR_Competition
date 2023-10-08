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
import cv2
import roar_py_interface

logging.basicConfig(level=logging.INFO)
matplotlib.use("TkAgg")


class PyGameViewer2:
    def __init__(
            self
    ):
        self.screen = None
        self.clock = None

    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        self.figure, self.ax = plt.subplots(nrows=1, ncols=1)
        self.lines = self.ax.plot(1, 50, 'bo', markersize=3)[0]
        self.depth_value_array = []
        self.seconds_array = []
        self.samesecond_array = []
        self.presec = 0

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
        # logging.infO("img_Pil:Depth = " + str(image_pil))
        image2_np = image2_np[100:150, len(image2_np) - 50: len(image2_np) + 50]
        # image2_np = np.log(image2_np)
        image2_np = np.clip(image2_np, 0, 40)
        min, max = np.min(image2_np), np.max(image2_np)
        min, max = 0, 40
        normalized_image = (image2_np) / (max - min)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        image2_pil = Image.fromarray(normalized_image, mode="L")
        # image2_pil = ImageOps.invert(image2_pil)
        if self.screen is None:
            self.init_pygame(image_pil.width + image2_pil.width, image_pil.height)
        mplstyle.use('fast')
        depth_value = np.average(image2_np)
        intdp = int(depth_value)
        depth_value_text = ImageDraw.Draw(image2_pil)
        depth_value_text.text((2, 2), str(depth_value), fill=0)
        ticks = pygame.time.get_ticks()
        seconds = int(ticks / 1000)
        logging.info("sec: " + str(seconds) + ", depth:" + str(intdp))
        if (seconds != self.presec):
            self.seconds_array.append(seconds)
            self.depth_value_array.append(depth_value)
            self.samesecond_array = []
        if (seconds > 0 and seconds == self.seconds_array[-1]):
            self.samesecond_array.append(depth_value)
            # print('dp appened')
        if (seconds > 0 and len(self.samesecond_array) > 0):
            self.samesecond_array.sort()
            print("same sec: " + str(self.samesecond_array))
            self.depth_value_array.append(self.samesecond_array[0])
            self.depth_value_array.append(self.samesecond_array[-1])
            self.seconds_array.append(seconds)
            self.seconds_array.append(seconds)
        # the code above lowk does not work
        # debug code
        # print("seconds arr:" + str(self.depth_value_array))
        # print("value arr" + str(self.seconds_array))
        display_sec_arry = self.seconds_array
        display_depth_arry = self.depth_value_array
        self.presec = seconds
        # if(len(display_sec_arry) > 30):
        #     display_sec_arry = []
        #     display_sec_arry = []
        self.lines.set_xdata(display_sec_arry)
        self.lines.set_ydata(display_depth_arry)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.plot(display_sec_arry, display_depth_arry)
        x_val = location.x
        y_val = location.y
        print(str(x_val) + " " + str(y_val))
        # img = cv2.imread('src\MajorMap.jpg', cv2.IMREAD_REDUCED_COLOR_2)
        # cv2.imshow('map', img)
        # cv2.waitKey(0)
        plt.show(block=False)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # cv2.destroyAllWindows()
                pygame.quit()
                return None

        combined_img_pil = Image.new('RGB', (image_pil.width + image2_pil.width, image_pil.height), (250, 250, 250))
        combined_img_pil.paste(image_pil, (0, 0))
        combined_img_pil.paste(image2_pil, (image_pil.width, 0))
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

