"""
Description: 
version: v1.0
Author: HTY
Date: 2023-01-20 17:49:10
"""

import pygame
import sys
import numpy as np

white = (255, 255, 255)
black = (0, 0, 0)
grey = (150, 150, 150)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
agent_size = 8, 8
screen_size = 420, 420
canvas_size = 400, 400

if __name__ == "__main__":
    pygame.init()
    pygame.display.init()

    screen = pygame.display.set_mode(screen_size)   # whole window
    screen.fill(grey)

    canvas = pygame.Surface(canvas_size)    # map
    canvas.fill(white)

    background = pygame.Surface(canvas_size)  # 原始map
    background.fill(white)

    # 生成一段路径
    init_pos = 50, 50
    target_pos = 300, 400
    points = []
    for x in np.arange(init_pos[0], target_pos[0]):
        points.append([x, x * (target_pos[1] - init_pos[1]) / (target_pos[0] - init_pos[0])])

    for point in points:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                sys.exit()

        canvas.blit(background, np.subtract(screen_size, canvas_size) / 2)  # 擦除画布
        pygame.draw.rect(canvas, blue, [point[0], point[1], agent_size[0], agent_size[1]])
        screen.blit(canvas, np.subtract(screen_size, canvas_size) / 2)  # 把画布上的内容更新到窗口中

        pygame.display.flip()

        pygame.time.delay(100)



