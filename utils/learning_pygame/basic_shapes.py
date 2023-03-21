"""
Description: 
version: v1.0
Author: HTY
Date: 2023-01-19 15:18:42
"""

import pygame
import sys

if __name__ == "__main__":
    size = width, height = 1000, 500
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    screen = pygame.display.set_mode(size)
    screen.fill(white)

    pygame.draw.line(surface=screen, color=red, start_pos=(60, 80), end_pos=(400, 200), width=2)
    pygame.draw.rect(surface=screen, color=green, rect=pygame.Rect(400, 200, 60, 60))
    pygame.draw.circle(surface=screen, color=blue, center=[20, 300], radius=20, width=0)    # width==0 means solid
    pygame.draw.polygon(surface=screen, color=black, points=[[20, 20], [20, 60], [60, 20]])

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                sys.exit()



