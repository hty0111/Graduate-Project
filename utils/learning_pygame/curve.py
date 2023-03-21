"""
Description: 
version: v1.0
Author: HTY
Date: 2023-01-19 16:36:34
"""

import pygame
import sys
import numpy as np

if __name__ == "__main__":
    size = width, height = 1000, 500
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    screen = pygame.display.set_mode(size)
    screen.fill(white)

    points = []
    for x in np.arange(0, 1000, 2):
        y = np.cos(x) * 10 + 200
        points.append([x, y])

    pygame.draw.aalines(surface=screen, color=black, closed=False, points=points)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                sys.exit()
