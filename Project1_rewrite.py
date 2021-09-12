
import numpy as np
import pygame
from math import cos, sin, pi, atan2

# ================================================================================= #
# ==============================    Item Classes     ============================== #
# ================================================================================= #

class circle_obj:
    def __init__(self, radius, initXY, initRot, color, phi, disp, disp_height):
        '''all coordinate are "true" coordinates in room frame'''
        self.radius = radius
        self.X = initXY[0]
        self.Y = initXY[1]
        self.R = initRot
        self.color = color
        self.phi = phi
        self.display = disp
        self.display_height = disp_height
    
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
    
    def theta2(R):
        ''''''
        return atan2(R[1][0], R[0][0])    
    
    def show_coordSys(self):
        ''''''
        x = self.X
        y = self.display_height-self.Y
        radius = self.radius
        
        theta = theta2(self.R)
        axis_1 = [[x,y],
                  [x+cos(theta)*radius*1.5, (y-sin(theta)*radius*1.5)]]
        axis_2 = [[x,y],
                  [x+cos(theta+pi/2)*radius*1.5, (y-sin(theta+pi/2)*radius*1.5)]]
        
        pygame.draw.polygon(self.display, (255,0,0), axis_1, width=1)
        pygame.draw.polygon(self.display, (255,0,0), axis_2, width=1)        
        return  

    def show_triangle(self):
        ''''''
        x = self.X
        y = self.display_height-self.Y
        radius = self.radius
        theta = theta2(self.R)
        
        points = [[x+cos(theta+self.phi+0)*radius, (y-sin(theta+self.phi+0)*radius)],
                  [x+cos(theta+self.phi+2*pi/3)*radius, (y-sin(theta+self.phi+2*pi/3)*radius)],
                  [x+cos(theta+self.phi+4*pi/3)*radius, (y-sin(theta+self.phi+4*pi/3)*radius)]]
        pygame.draw.polygon(self.display, (255,0,0), points, width=1)
        return
    
    def draw(self):
        ''''''
        pygame.draw.circle(self.display, self.color, (self.X, self.display_height-self.Y), self.radius, width=1)    
        self.show_triangle()
        self.show_coordSys()

class rect_obj:
    def __init__(self, width, height, initXY, initRot, color, disp, disp_height):
        self.width = width
        self.height = height
        self.X = initXY[0]
        self.Y = initXY[1]
        self.R = initRot
        self.color = color
        self.radius = min(width, height)
        self.display = disp
        self.display_height = disp_height

    def theta2(R):
        ''''''
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

    def show_coordSys(self):
        ''''''
        x = self.X
        y = self.display_height-self.Y
        radius = self.radius
        
        theta = theta2(self.R)
        axis_1 = [[x,y],
                  [x+cos(theta)*radius, (y-sin(theta)*radius)]]
        axis_2 = [[x,y],
                  [x+cos(theta+pi/2)*radius, (y-sin(theta+pi/2)*radius)]]
        
        pygame.draw.polygon(self.display, (255,0,0), axis_1, width=1)
        pygame.draw.polygon(self.display, (255,0,0), axis_2, width=1)        
        return  

    def draw(self):
        ''''''
        rect = pygame.Rect((self.X-self.width/2),
                         self.display_height - (self.Y+self.height/2),
                         self.width, self.height)
        pygame.draw.rect(self.display, self.color, rect, width=1)
        self.show_coordSys()

# ================================================================================= #
# ============================    Helpful Functions    ============================ #
# ================================================================================= #

def theta2(R):
        ''''''
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




# ================================================================================= #
# ============================    Project 1 Sections    =========================== #
# ================================================================================= #

def run_part1():
    # part a
    r10 = np.array([[0,1],[-1,0]])
    p10 = np.array([[-0.5],[4]])
    r01 = np.transpose(r10)
    p01 = r10 @ p10

    # part b
    r02 = rot2(0)
    p21 = np.array([[3],[0]])
    p02 = r02 @ (p01) - p21

    # part c
    r03 = rot2(pi/4)
    p03 = np.array([[2.5],[2.5]])

    # part d
    r04 = rot2(-pi/2)
    p24 = np.array([[0],[3.5]])
    p04 = p02 + p24
    
    #print(r01)
    #print(p01)
    ##print(r02)
    #print(p02)
    #print(r03)
    #print(p03)
    #print(r04)
    #print(p04)

    # plotting (showing figure)
    display_width = 500
    display_height = 500
    display_scale = 100

    pygame.init()

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project Part 1')

    black = (0,0,0)
    white = (255, 255, 255)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    light_blue = (0,255,255)
    gray = (225,225,225)
    brown = (130,90,20)
    yellow = (130,124,20)

    r_radius = 0.3 * display_scale
    p_radius = 0.2 * display_scale
    table_dim = [0.5*display_scale, 0.5*display_scale]
    shelf_dim = [0.3*display_scale, 0.8*display_scale]

    p01 *= display_scale
    p02 *= display_scale
    p03 *= display_scale
    p04 *= display_scale

    robot = circle_obj(r_radius, p01, r01, blue, 0, gameDisplay, display_height)
    person = circle_obj(p_radius, p02, r02, green, 0, gameDisplay, display_height)
    table = rect_obj(table_dim[0], table_dim[1], p03, r03, brown, gameDisplay, display_height)
    shelf = rect_obj(shelf_dim[0], shelf_dim[1], p04, r04, yellow, gameDisplay, display_height)

    def draw_grid(num_blocks=2):
        for i in range(0,display_width,int(display_scale/num_blocks)):
            for j in range(0,display_height,int(display_scale/num_blocks)):
                rect = pygame.Rect(i, j, int(display_scale/num_blocks), int(display_scale/num_blocks))
                pygame.draw.rect(gameDisplay, white, rect, 1)


    crashed = False
    while not crashed:
        gameDisplay.fill(gray)
        draw_grid()
        robot.draw()
        person.draw()
        table.draw()
        shelf.draw()
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(gameDisplay, 'part1.png') 
                crashed = True
            #print(event)

        pygame.display.update()
    
    pygame.quit()
    quit()

def run_part2():

    p02 = np.array([[1],[0.5]])
    r02 = np.array([[1,0],[0,1]])

    p03 = np.array([[2.5],[2.5]])
    r03 = np.array([[cos(pi/4),-sin(pi/4)],[sin(pi/4),cos(pi/4)]])

    p04 = np.array([[1],[4]])
    r04 = np.array([[0,1],[-1,0]])

    #part a
    r01a = np.array([[-1,0],[0,-1]])
    p01a = p04 + np.array([[0.1],[0]]) + np.array([[0.3/2],[0]]) + np.array([[0.3],[0]])
    #print(p01a)

    #part b
    r01b = np.array([[0,1],[-1,0]])
    p01b = p02 + np.array([[0],[0.1]]) + np.array([[0],[0.2]]) + np.array([[0],[0.3]])
    #print(p01b)
    
    #print(r01)
    #print(p01)
    #print(r02)
    #print(p02)
    #print(r03)
    #print(p03)
    #print(r04)
    #print(p04)

    # plotting (showing figure)
    display_width = 500
    display_height = 500
    display_scale = 100

    pygame.init()

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project Part 1')

    black = (0,0,0)
    white = (255, 255, 255)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    light_blue = (0,200,255)
    gray = (225,225,225)
    brown = (130,90,20)
    yellow = (130,124,20)

    r_radius = 0.3 * display_scale
    p_radius = 0.2 * display_scale
    table_dim = [0.5*display_scale, 0.5*display_scale]
    shelf_dim = [0.3*display_scale, 0.8*display_scale]

    p01a *= display_scale
    p01b *= display_scale
    p02 *= display_scale
    p03 *= display_scale
    p04 *= display_scale

    robot1 = circle_obj(r_radius, p01a, r01a, blue, 0, gameDisplay, display_height)
    robot2 = circle_obj(r_radius, p01b, r01b, light_blue, 0, gameDisplay, display_height)
    person = circle_obj(p_radius, p02, r02, green, 0, gameDisplay, display_height)
    table = rect_obj(table_dim[0], table_dim[1], p03, r03, brown, gameDisplay, display_height)
    shelf = rect_obj(shelf_dim[0], shelf_dim[1], p04, r04, yellow, gameDisplay, display_height)

    def draw_grid(num_blocks=2):
        for i in range(0,display_width,int(display_scale/num_blocks)):
            for j in range(0,display_height,int(display_scale/num_blocks)):
                rect = pygame.Rect(i, j, int(display_scale/num_blocks), int(display_scale/num_blocks))
                pygame.draw.rect(gameDisplay, white, rect, 1)


    crashed = False
    while not crashed:
        gameDisplay.fill(gray)
        draw_grid()
        robot1.draw()
        robot2.draw()
        person.draw()
        table.draw()
        shelf.draw()
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(gameDisplay, 'part2.png') 
                crashed = True
            #print(event)

        pygame.display.update()
    
    pygame.quit()
    quit()

def run_part3b():

    p02 = np.array([[1],[0.5]])
    r02 = np.array([[1,0],[0,1]])

    p03 = np.array([[2.5],[2.5]])
    r03 = np.array([[cos(pi/4),-sin(pi/4)],[sin(pi/4),cos(pi/4)]])

    p04 = np.array([[1],[4]])
    r04 = np.array([[0,1],[-1,0]])

    # initial_pose:
    r01 = np.array([[0,-1],[1,0]])
    p01 = np.array([[4],[0.5]])

    # target_pose 1:
    r01a = np.array([[-1,0],[0,-1]])
    p01a = p04 + np.array([[0.1],[0]]) + np.array([[0.3/2],[0]]) + np.array([[0.3],[0]])
    # target_pose 2:
    r01b = np.array([[0,1],[-1,0]])
    p01b = p02 + np.array([[0],[0.1]]) + np.array([[0],[0.2]]) + np.array([[0],[0.3]])

    # path break pose 1:
    p01k1 = p03 + np.array([[2.5/2-cos(5*pi/4)],[2.5/2-sin(5*pi/4]])
    # path break pose 2:
    p01k2 = p04 + np.array([[0.3/2-cos(3*pi/4)],[0.8/2+sin(3*pi/4]])
    
    




# ================================================================================= #
# =========================    Execute Project Sections    ======================== #
# ================================================================================= #


#run_part1()
#run_part2()
#run_part3b()