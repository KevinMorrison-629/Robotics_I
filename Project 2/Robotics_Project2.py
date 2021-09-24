
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import os
from math import cos, sin, pi, atan2
import pandas as pd

sPath = pd.read_excel('S_path.xlsx')
sPath = sPath.to_numpy().T




# ================================================================================= #
# ============================    Helpful Functions    ============================ #
# ================================================================================= #

def theta2(R):
        '''calculates the 1d rotation based on a 2x2 rotation matrix

        Inputs
            R : 2x2 rotation matrix
        Ouputs
            theta : rotation corresponding to the rotation matrix
        '''
        if atan2(R[1][0], R[0][0]) < 0:
            return atan2(R[1][0], R[0][0]) + 2*pi
        else:
            return atan2(R[1][0], R[0][0])  

def rot2(theta):
        '''creates a 2x2 rotation matrix given a specified angle, theta
        
        Inputs
             theta : the angle at which the object is rotated
        Outputs
             R : the rotation matrix specified by the input angle theta'''
        
        c = cos(theta)
        s = sin(theta)
        R = np.array([[c, -s],[s, c]])
        return R

def rot3(theta):
    '''creates a 3x3 rotation matrix given a specified angle, theta
        
    Inputs
            theta : the angle at which the object is rotated
    Outputs
            R : the rotation matrix specified by the input angle theta'''
        
    c = cos(theta)
    s = sin(theta)
    R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])
    return R

# ================================================================================= #
# ==============================    Item Classes     ============================== #
# ================================================================================= #


class path_obj:
    def __init__(self, points, isNormalOutward, color, disp, disp_height):
        self.points = points
        scale = disp_height/4
        self.points = [[(each[0]+1)*scale,disp_height-(each[1]+2)*scale] for each in self.points]
        self.color = color
        self.disp = disp
        self.isNormalOutward = isNormalOutward

        self.path_normals = []
        for i in range(len(self.points)-1):
            point1 = self.points[i]
            point2 = self.points[i+1]
            if isNormalOutward:
                m = [(point2[1]-point1[1]),(point1[0]-point2[0])]
            else:
                m = [(point1[1]-point2[1]),(point2[0]-point1[0])]
            center = [point1[0] + (point2[0]-point1[0])/2 , point1[1] + (point2[1]-point1[1])/2]
            self.path_normals.append([center,[sum(x) for x in zip(center, m)]])

    def show_path(self):
        for i in range(len(self.points)-1):
            pygame.draw.line(self.disp, self.color, self.points[i], self.points[i+1], width=1)
    
    def show_path_normal(self):
        for i in range(len(self.path_normals)):
            pygame.draw.line(self.disp, self.color, self.path_normals[i][0], self.path_normals[i][1], width=1)
    
    def switch_normals(self):

        if self.isNormalOutward:
            self.isNormalOutward = False
        else:
            self.isNormalOutward = True

        self.path_normals = []
        for i in range(len(self.points)-1):
            point1 = self.points[i]
            point2 = self.points[i+1]
            if self.isNormalOutward:
                m = [(point2[1]-point1[1]),(point1[0]-point2[0])]
            else:
                m = [(point1[1]-point2[1]),(point2[0]-point1[0])]
            center = [point1[0] + (point2[0]-point1[0])/2 , point1[1] + (point2[1]-point1[1])/2]
            self.path_normals.append([center,[sum(x) for x in zip(center, m)]])


class robot:
    def __init__(self, p0, l, wmax, disp, colors):
        ''''''
        self.p0 = p0
        self.l = l
        self.wmax = wmax
        self.disp = disp
        self.colors = colors

        # assume zero configuration is a fully extended arm
        # self.T is the homogeneous transform for the pose
        #self.T, self.J = forward_kinematics(np.zeros(3))

    def forward_kinematics(self, q):

        self.T = np.zeros((4,4))
        self.J = np.zeros((3,len(q)))

        R0T = rot2(sum(q))
        p0T = self.p0
        for i in range(len(q)):
            p0T = np.add(p0T, (rot2(sum(q[0:i])) @ np.array([[self.l[i]],[0]])))
        self.T = np.zeros((4,4))
        self.T[0:2,0:2] = R0T
        self.T[0,3] = p0T[0][0]
        self.T[1,3] = p0T[1][0]
        self.T[2,2] = 1
        self.T[3,3] = 1

        for i in range(len(q)):
            R0ipiT = np.zeros((2,1))
            for j in range(i,len(q)):
                R0ipiT += rot2(sum(q[i:j])) @ np.array([[self.l[j]],[0]])
            self.J[0,i] = R0ipiT[0][0]
            self.J[1,i] = R0ipiT[1][0]
        print(self.T)
        print(self.J)
        return self.T, self.J










    def inverse_kinematics(self, xT, yT):
        
        return q

    def draw_robot(self):
        points = []
        for i in range(len(self.links)):
            points.append([sum(self.links[0,0:i]),sum(self.links[1,0:i])])
        for i in range(len(points)-1):
            pygame.draw.line(self.disp, self.colors[i], points[i], points[i+1], width=1)





colors = [(255,0,0),(0,255,0),(0,0,255)] # RGB
p0 = np.array([[0],[0]])
l = np.array([1.5,1.5,0.5,3.5])
umax = 1
wmax = 1
q = np.array([0, 0, 0, pi/2])








# plotting (showing figure)
display_width = 1000
display_height = 1000
display_scale = 100
gray = (200,200,200)

pygame.init()
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Project 2')


robot1 = robot(p0, l, wmax, gameDisplay, colors)
T, J = robot1.forward_kinematics(q)

path1 = path_obj(sPath, False, [255,255,255], gameDisplay, display_height)


crashed = False
while not crashed:
    time.sleep(3)
    path1.switch_normals()
    gameDisplay.fill(gray)
    path1.show_path()
    path1.show_path_normal()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #pygame.image.save(gameDisplay, 'part1.png') 
            crashed = True
    pygame.display.update()
pygame.quit()


