#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:09:37 2022

@author: maxcai
"""

import numpy as np
import pygame
import sys

#dd_theta + g/l sin(theta) = 0

g = 10
l = 10
m = 1
dt = 1E-3
theta0 = 3.1
omega0 = 0
t = np.arange(0, 40, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
energy = np.zeros_like(t)
theta[0] = theta0
omega[0] = omega0

#Euler Forward
for i in range(t.shape[0]-1):
    theta[i+1] = theta[i] + dt * omega[i]
    omega[i+1] = omega[i] - g/l * dt * np.sin(theta[i])

energy = 0.5 * m * np.square(omega) * l**2 - (m*g*l*np.cos(theta))
print("Energy change = ", energy[-1] - energy[0])





class Pendulum(pygame.sprite.Sprite):
    def __init__(self,window):
        super().__init__()
        self.window = window
        print("Successfully created pendulum")
        #self.image = pygame.image.load("frog/attack_1.gif")
        #self.rect = self.image.get_rect()
        #self.rect.topleft = [200,200]
    def update(self,theta):
        pygame.draw.circle(self.window, (255,0,0), (200,200),50)
        #print(theta, "is the theta given to pendulum")
        pass
        
        

# General setup
pygame.init()
clock = pygame.time.Clock()

# Game Screen
screen_width = 400
screen_height = 400
screen = pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption("Pendulum Animation")

# Creating the sprites and groups
# moving_sprites = pygame.sprite.Group()
pendulum1 = Pendulum(screen)
# moving_sprites.add(pendulum1)

while True:
    for event in pygame.event.get():   
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    pendulum1.update(10)
    #pendulum1.draw()
    pygame.display.flip()
    
            