"""
Description: 
version: v1.0
Author: HTY
Date: 2023-01-17 23:33:39
"""

import pygame
import sys

if __name__ == "__main__":
    pygame.init()

    ball = pygame.image.load("magent-graph-1.gif")
    ball_rect = ball.get_rect()

    size = width, height = int(ball.get_width() * 1.5), int(ball.get_height() * 1.5)
    speed = [1, 1]
    black = 0, 0, 0

    screen = pygame.display.set_mode(size)

    while True:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                sys.exit()

        ball_rect = ball_rect.move(speed)

        if ball_rect.left < 0 or ball_rect.right > width:
            speed[0] = -speed[0]
        if ball_rect.top < 0 or ball_rect.bottom > height:
            speed[1] = -speed[1]

        screen.fill(black)
        screen.blit(ball, ball_rect)
        pygame.display.flip()

