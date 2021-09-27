
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import os
from math import *
import pandas as pd

sPath = pd.read_excel('S_path.xlsx')
sPath = sPath.to_numpy().T
sPath = np.array([[(each[0]+1),(each[1]+2)] for each in sPath])



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

def vec_angle(u1, u2):
    '''Calculates the angle theta between two vectors
    
    Inputs
        u1 : the first vector
        u2 : the second vector
    Outputs
        theta : the angle between u1 and u2 in the counter-clockwise direction'''
    k = np.array([0,0,1])
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)

    theta = 2*atan2(np.linalg.norm(u1-u2),np.linalg.norm(u1+u2))

    print(k.T)
    print(np.cross(u1, u2))

    if k.T @ np.cross(u1, u2) < 0:
        theta = theta*-1

    return theta

def circ_int(x0, y0, r0, x1, y1, r1):
    '''Calculates the intersection between two circles
    
    Inputs
        a : x-coordinate for first circle
        b : y-coordinate for first circle
        r0 : radius for first circle
        c : x-coordinate for second circle
        d : y-coordinate for second circle
        r1 : radius for second circle
    Ouputs
        [points1, point2] : two valid points of intersection between circles'''
    
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist > r0 + r1 or dist < abs(r0-r1):
        return []
    if abs(dist)-0.001 < 0 and abs(r0-r1) <0.001:
        return []
    
    a = (r0**2 - r1**2 + dist**2)/(2*dist)
    h = np.sqrt(r0**2 - a**2)

    x2 = x0 + a*(x1-x0)/dist
    y2 = y0 + a*(y1-y0)/dist

    x3_1 = x2 + h*(y1-y0)/dist
    y3_1 = y2 - h*(x1-x0)/dist

    x3_2 = x2 - h*(y1-y0)/dist
    y3_2 = y2 + h*(x1-x0)/dist
    

    return [[x3_1, y3_1],[x3_2, y3_2]]


    #dist = np.sqrt((c-a)**2 + (d-b)**2)
    #if r0+r1 > dist and dist > abs(r0-r1):
    #    delta = 0.25*np.sqrt((dist+r0+r1)*(dist+r0-r1)*(dist-r0+r1)*(-dist+r0+r1))

    #    x1 = ((a+c)/2) + (c-a)*(r0**2 - r1**2)/(2*(dist**2)) + 2*delta*(b-d)/(dist**2)
    #    x2 = ((a+c)/2) + (c-a)*(r0**2 - r1**2)/(2*(dist**2)) - 2*delta*(b-d)/(dist**2)

    #    y1 = ((b+d)/2) + (d-b)*(r0**2 - r1**2)/(2*(dist**2)) + 2*delta*(a-c)/(dist**2)
    #    y2 = ((b+d)/2) + (d-b)*(r0**2 - r1**2)/(2*(dist**2)) - 2*delta*(a-c)/(dist**2)

    #    return [(x1, y1), (x2, y2)]
    #else:
    #    return []

def create_T(qT, pT):
    T = np.eye(4)
    T[0:2,0:2] = rot2(qT)
    T[0:2,3] = np.reshape(pT,(2,))
    return T

# ================================================================================= #
# ==============================    Item Classes     ============================== #
# ================================================================================= #


class path:
    def __init__(self, points, disp, disp_height, color):
        ''''''
        self.points = points
        self.disp = disp
        self.disp_height = disp_height
        self.color = color

        self.scale = disp_height/4

        self.calc_normals()
        self.calc_orientations()
    
    def calc_normals(self, outward=True):
        ''''''
        self.normals = []
        for i in range(len(self.points)-1):
            if outward:
                m = [(self.points[i+1][1]-self.points[i][1]),(self.points[i][0]-self.points[i+1][0])]
            else:
                m = [(self.points[i][1]-self.points[i+1][1]),(self.points[i+1][0]-self.points[i][0])]
            self.normals.append([self.points[i], [sum(x) for x in zip(self.points[i],m)]])
        self.normals.append([self.points[-1], [sum(x) for x in zip(self.points[-1],m)]])

    def calc_orientations(self):
        ''''''
        self.path_orientations = np.array([])
        for each in self.normals:
            u1 = np.array([[1],[0],[0]])
            u2 = np.array(each[1]) - np.array(each[0])
            u2 = np.reshape(u2,(1,2))[0]
            u2 = np.append(u2, 0)
            u1 = np.reshape(u1,(1,3))[0]
            self.path_orientations = np.append(self.path_orientations, vec_angle(u1, u2))
    
    def show(self, showNorms=True):
        ''''''
        scaled_points = [[(each[0])*self.scale,self.disp_height-(each[1])*self.scale] for each in self.points]
        for i in range(len(self.points)-1):
            pygame.draw.line(self.disp, self.color, scaled_points[i], scaled_points[i+1], width=1)
        if showNorms:
            scaled_normals = [[[each[0]*self.scale, self.disp_height-each[1]*self.scale] for each in points] for points in self.normals]
            for i in range(len(self.points)):
                pygame.draw.line(self.disp, self.color, scaled_normals[i][0], scaled_normals[i][1], width=1)

class robot:
    def __init__(self, p0, l_list, wmax, elbowUp, disp, disp_height, colors):
        ''''''
        self.p0 = p0
        self.l_list = l_list
        self.wmax = wmax
        self.elbowUp = elbowUp
        self.disp = disp
        self.disp_height = disp_height
        self.colors = colors

        self.scale = disp_height / 4

        self.T, self.J = self.fk(np.zeros((3,)))



    def fk(self, q):
        ''''''
        self.links = [self.p0]

        R0T = rot2(sum(q))
        T = np.eye(4)
        p0T = self.p0.copy()
        # calculate the arm pose
        for i in range(len(q)):
            link_vec = rot2(sum(q[0:i+1])) @ np.array([[self.l_list[i]],[0]])
            p0T = np.add(p0T, link_vec)
            self.links.append(self.links[-1] + link_vec)
        T[0:2, 0:2] = R0T
        T[0:2,3] = np.reshape(p0T,(2,))
        # calculate the 3-arm Jacobian
        Jac = np.ones((3,len(q)))
        for i in range(len(q)):
            R0ipiT = np.zeros((2,1))
            for j in range(i,len(q)):
                R0ipiT += rot2(sum(q[i:j+1])) @ np.array([[self.l_list[j]],[0]])
            Jac[0,i] = R0ipiT[0][0]
            Jac[1,i] = R0ipiT[1][0]
        return T, Jac

    def geom_IK(self, T):
        ''''''
        qT = theta2(T[0:2, 0:2])
        pT = T[0:2,3]
        
        x0, y0 = self.p0[0], self.p0[1]
        xT, yT = pT[0], pT[1]

        l1, l2, l3 = self.l_list[0], self.l_list[1], self.l_list[2]

        x3 = xT - l3*cos(qT)
        y3 = yT - l3*sin(qT)

        points = circ_int(x0, y0, l1, x3, y3, l2)
        if self.elbowUp:
            x2 = points[0][0]
            y2 = points[0][1]
        else:
            x2 = points[1][0]
            y2 = points[1][1]

        print(x3, y3)

        test_points = np.array([[xT*self.scale, self.disp_height-yT*self.scale],
                                [x0*self.scale, self.disp_height-y0*self.scale],
                                [x2*self.scale, self.disp_height-y2*self.scale],
                                [x3*self.scale, self.disp_height-y3*self.scale]])
        for each in test_points:
            pygame.draw.circle(self.disp, [255,0,255], each, 5, width=1)

        u1 = np.array([1,0,0])
        u2 = np.array([x2-x0, y2-y0])
        u2 = np.append(u2, 0)
        q1 = vec_angle(u1, u2)

        u3 = np.array([x3-x2, y3-y2])
        u3 = np.append(u3, 0)
        q2 = vec_angle(u2, u3)

        u4 = np.array([xT-x3, yT-y3])
        u4 = np.append(u4, 0)
        q3 = vec_angle(u3, u4)

        #print(np.array([q1, q2, q3])/pi)

        #q1 = atan2(y2-y0, x2-x0)
        #q2 = atan2(y3-y2, x3-x2)
        #q3 = atan2(yT-y3, xT-x3)
        self.fk(np.array([q1, q2, q3]))

        return np.array([q1, q2, q3])

    def show(self):
        ''''''
        link_points = self.links.copy()
        link_points = [[each[0]*self.scale, each[1]*self.scale] for each in link_points]
        link_points = [[each[0][0], self.disp_height-each[1][0]] for each in link_points]

        pygame.draw.circle(self.disp, [0,0,0], link_points[0], 5, width=1)
        pygame.draw.circle(self.disp, [255,255,255], link_points[-1], 5, width=1)
        for i in range(len(self.links)-1):
            pygame.draw.line(self.disp, self.colors[i%3], link_points[i], link_points[i+1], width=1)



if __name__ == '__main__':
    colors = [(0,0,255),(255,0,0),(0,255,0)] # RGB
    p0 = np.array([[1.5],[2]])
    l_list = np.array([1.5,1.5,0.5])
    wmax = 1
    
    # plotting (showing figure)
    display_width = 1000
    display_height = 1000
    display_scale = 100
    gray = (200,200,200)
    pygame.init()
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project 2')


    path1 = path(sPath, gameDisplay, display_height, [0,0,255])
    robot1 = robot(p0, l_list, wmax, True, gameDisplay, display_height, colors)
    robot2 = robot(p0, l_list, wmax, False, gameDisplay, display_height, colors)








    # Pygame Show Simulation:

    crashed = False
    pose_num = 0

    while not crashed and pose_num < 101:
        gameDisplay.fill(gray)
        path1.show()

        qT = path1.path_orientations[pose_num]
        pT = path1.points[pose_num]
        next_T = create_T(qT, pT)

        next_qs = robot1.geom_IK(next_T)
        robot1.show()
        next_qs = robot2.geom_IK(next_T)
        robot2.show()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #pygame.image.save(gameDisplay, 'part1.png') 
                crashed = True
        pygame.display.update()
        time.sleep(0.1)
        pose_num += 1
    pygame.quit()





