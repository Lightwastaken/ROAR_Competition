import pygame
import logging
logging.basicConfig(level=logging.INFO)

class checkpoints:
    def __init__(self):
        pygame.init()
        self.lap_count = 0
        self.clock = pygame.time.Clock()
        self.in_start_box = False
        self.is_finished = False
        self.checkpoint_list = [((-240, -220), (770, 810)), ((385, 905), (960, 1000)), ((720, 780), (1010, 1030)), ((730, 750), (700, 740)),
                                ((-10, 10), (80, 120)), ((-95, -55),(-110, -90)), ((-170, -150), (-1070, -1030)), ((-390, -350), (-820, -800)),
                                ((-360, -320), (220, 240))]
        self.checkpoint_index = 0
        self.f = open('checkpointlog.txt', 'a+')
        contents = self.f.readlines()
        a = contents.count("~")
        self.f.write("\n-------------------------------------\n")
        self.f.write("attempt: " + str(a) + '~')




    def update_checkpoints(self, x_coord, y_coord):
        ticks = pygame.time.get_ticks()
        seconds = int(ticks / 1000)

        if self.checkpoint_index < len(self.checkpoint_list):
            x_coord_min = self.checkpoint_list[self.checkpoint_index][0][0]
            x_coord_max = self.checkpoint_list[self.checkpoint_index][0][1]

            y_coord_min = self.checkpoint_list[self.checkpoint_index][1][0]
            y_coord_max = self.checkpoint_list[self.checkpoint_index][1][1]
        if -300 < x_coord < -290 and 420 < y_coord < 430 and not self.in_start_box and not self.is_finished:
            self.in_start_box = True
            self.lap_count += 1
            if self.lap_count == 4:
                print("Total time:", seconds)
                self.f.write("Total time: " + str(seconds))
                self.is_finished = True

            else:
                print("Lap", self.lap_count, ":", seconds)
                self.f.write("\nLap " + str(self.lap_count) + ": " + str(seconds) + "; ")
                self.checkpoint_index = 0

        elif  self.checkpoint_index < len(self.checkpoint_list) and x_coord_min < x_coord < x_coord_max and y_coord_min < y_coord < y_coord_max and not self.is_finished:
            self.in_start_box = False
            print("checkpoint", self.checkpoint_index + 1, ":", seconds)
            self.f.write("checkpoint " + str(self.checkpoint_index + 1) + ": " + str(seconds) + "; ")
            self.checkpoint_index += 1

        # this doesn't work
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                print("quit")
                self.f.write("\n---------------------------------------\n")
                self.f.close()
                pygame.quit()
        self.clock.tick(60)
