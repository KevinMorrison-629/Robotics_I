
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import os
from math import cos, sin, pi, atan2




# ================================================================================= #
# ============================    Helpful Functions    ============================ #
# ================================================================================= #

def theta2(R):
        ''''''
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

def create_position_path(p0, p1, N):
    l_slp = np.linalg.norm(p1-p0)
    lambda_vec = np.linspace(0, l_slp, N)
    path = [p0*(1-(each/l_slp)) + p1*(each/l_slp) for each in lambda_vec]
    return lambda_vec, path

def create_orientation_path(r0, r1, lambda_vec):
    theta0 = theta2(r0)
    theta1 = theta2(r1)

    if theta1 < 0:
        theta1 += 2*pi

    rot_poses = np.linspace(theta0, theta1, len(lambda_vec))
    return [rot2(each) for each in rot_poses]


def calc_time(p0, p1, theta0, theta1, umax, wmax):
    pos_path_prime = (p1-p0)/ np.linalg.norm(p1-p0)
    rot_path_prime = (theta1-theta0)/ np.linalg.norm(p1-p0)

    ldot_list = [each[0] for each in abs(umax/pos_path_prime)]
    ldot_list.append(abs(wmax/rot_path_prime))

    ldot_max = min(ldot_list)
    T = np.linalg.norm(p1-p0)/ldot_max
    return T, pos_path_prime*ldot_max, rot_path_prime*ldot_max



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
        self.radius = ((width/2)**2 + (height/2)**2)**(1/2)
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
        theta = theta2(self.R)
        radius = self.radius
        x = self.X
        y = self.display_height-self.Y

        points = []
        w = self.width
        h = self.height
        points.append([x+radius*cos(np.arctan2(w,h)+theta), y-radius*sin(np.arctan2(w,h)+theta)])
        points.append([x+radius*cos(np.arctan2(-w,h)+theta), y-radius*sin(np.arctan2(-w,h)+theta)])
        points.append([x+radius*cos(np.arctan2(-w,-h)+theta), y-radius*sin(np.arctan2(-w,-h)+theta)])
        points.append([x+radius*cos(np.arctan2(w,-h)+theta), y-radius*sin(np.arctan2(w,-h)+theta)])
        pygame.draw.polygon(self.display, self.color, points, width=1)

        self.show_coordSys()

class path_obj:
    def __init__(self, points, color, disp, disp_height):
        self.points = points
        self.points = [[each[0],disp_height-each[1]] for each in self.points]
        self.color = color
        self.disp = disp

    def show_path(self):
        for i in range(len(self.points)-1):
            #print(self.points[i])
            #print(self.points[i+1])
            #print('=====\n')
            pygame.draw.line(self.disp, self.color, self.points[i], self.points[i+1], width=1)



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
    
    print('r01:')
    print(r01)
    print('p01:')
    print(p01)
    print('r02:')
    print(r02)
    print('p02:')
    print(p02)
    print('r03:')
    print(r03)
    print('p03:')
    print(p03)
    print('r04:')
    print(r04)
    print('p04:')
    print(p04)

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

def run_part2():

    p02 = np.array([[1],[0.5]])
    r02 = np.array([[1,0],[0,1]])

    p03 = np.array([[2.5],[2.5]])
    r03 = rot2(pi/4)

    p04 = np.array([[1],[4]])
    r04 = np.array([[0,1],[-1,0]])

    #part a
    r01a = np.array([[-1,0],[0,-1]])
    p01a = p04 + np.array([[0.1],[0]]) + np.array([[0.3/2],[0]]) + np.array([[0.3],[0]])
    print('p01a:')
    print(p01a)
    print('r01a:')
    print(r01a)

    #part b
    r01b = np.array([[0,1],[-1,0]])
    p01b = p02 + np.array([[0],[0.1]]) + np.array([[0],[0.2]]) + np.array([[0],[0.3]])
    print('p01b:')
    print(p01b)
    print('r01b:')
    print(r01b)

    # plotting (showing figure)
    display_width = 500
    display_height = 500
    display_scale = 100

    pygame.init()

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project Part 2')

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

def run_part3b():

    p02 = np.array([[1],[0.5]])
    r02 = np.array([[1,0],[0,1]])

    p03 = np.array([[2.5],[2.5]])
    r03 = rot2(pi/4)

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

    # this point is chosen by generating two lines originiating from the initial pose and the
    # pose_a, with the robot radius just in contact with the edge of the table. The intersection
    # of these two lines is the intermediary point, p01k1
    # path break pose 1:
    m1 = (0.5-2.5)/((4-0.3)-(2.5+np.sqrt(0.125)))
    m2 = ((4-0.3)-(2.5+np.sqrt(0.125)))/(1-2.5)
    x1 = 4
    y1 = 0.5
    x2 = 1
    y2 = 4
    x_tar = (m1*x1 - m2*x2 + y2 - y1)/(m1-m2)
    y_tar = m2*(x_tar-x2)+y2

    p01k1 = np.array([[x_tar],[y_tar]])
    #print(p01k1)

    # ==========================================
    N=24
    # path1
    lambda1, path1 = create_position_path(p01, p01k1, N)

    # path2
    lambda2, path2 = create_position_path(p01k1, p01a, N)
    lambda2 += lambda1[-1]
    # path3
    lambda3, path3 = create_position_path(p01a, p01b, N)
    lambda3 += lambda2[-1]

    lambda_vec = np.concatenate((lambda1, lambda2, lambda3))

    print('path length:',lambda_vec[-1])

    path = np.concatenate((path1, path2, path3))


    rot_path1 = create_orientation_path(r01, r01a, np.concatenate((lambda1, lambda2)))
    rot_path2 = create_orientation_path(r01a, r01b, lambda3)
    rot_path = np.concatenate((rot_path1, rot_path2))
    # ==========================================


    # plotting
    fig, ax = plt.subplots(3)

    ax[0].plot(lambda_vec, path[:,0], 'r-')
    ax[1].plot(lambda_vec, path[:,1], 'r-')
    ax[2].plot(lambda_vec, [theta2(each) for each in rot_path])

    ax[0].set_title('lambda vs. x-position (m)')
    ax[1].set_title('lambda vs. y-position (m)')
    ax[2].set_title('lambda vs. rotation (rad)')

    plt.setp(ax[0], ylabel='x-position (m)', xlabel='lambda')
    plt.setp(ax[1], ylabel='y-position (m)', xlabel='lambda')
    plt.setp(ax[2], ylabel='rotation (rad)', xlabel='lambda')

    fig.tight_layout()
    fig.savefig('part3b1.png')
    #plt.show()
    plt.close()
    
    # show in room

    # plotting (showing figure)
    display_width = 500
    display_height = 500
    display_scale = 100

    pygame.init()

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project Part 3')

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

    p01 *= display_scale
    p01k1 *= display_scale

    robot_list = []

    for i in range(len(lambda_vec)):
        robot_list.append(circle_obj(r_radius, path[i,:]*display_scale, rot_path[i], blue, 0, gameDisplay, display_height))

    robot1 = circle_obj(r_radius, p01a, r01a, blue, 0, gameDisplay, display_height)
    robot2 = circle_obj(r_radius, p01b, r01b, light_blue, 0, gameDisplay, display_height)
    person = circle_obj(p_radius, p02, r02, green, 0, gameDisplay, display_height)
    table = rect_obj(table_dim[0], table_dim[1], p03, r03, brown, gameDisplay, display_height)
    shelf = rect_obj(shelf_dim[0], shelf_dim[1], p04, r04, yellow, gameDisplay, display_height)

    path_points = [p01, p01k1, p01a, p01b]
    full_path = path_obj(path_points, green, gameDisplay, display_height)

    def draw_grid(num_blocks=2):
        for i in range(0,display_width,int(display_scale/num_blocks)):
            for j in range(0,display_height,int(display_scale/num_blocks)):
                rect = pygame.Rect(i, j, int(display_scale/num_blocks), int(display_scale/num_blocks))
                pygame.draw.rect(gameDisplay, white, rect, 1)
    

    crashed = False
    gameDisplay.fill(gray)
    draw_grid()
    while not crashed:

        for i in range(len(robot_list)):
            if i % 4 == 0:
                robot_list[i].draw()

        robot1.draw()
        robot2.draw()
        person.draw()
        table.draw()
        shelf.draw()
        full_path.show_path()
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(gameDisplay, 'part3b.png') 
                crashed = True
            #print(event)

        pygame.display.update()
    
    pygame.quit()

def run_part3c():

    p02 = np.array([[1],[0.5]])
    r02 = np.array([[1,0],[0,1]])

    p03 = np.array([[2.5],[2.5]])
    r03 = rot2(pi/4)

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
    m1 = (0.5-2.5)/((4-0.3)-(2.5+np.sqrt(0.125)))
    m2 = ((4-0.3)-(2.5+np.sqrt(0.125)))/(1-2.5)
    x1 = 4
    y1 = 0.5
    x2 = 1
    y2 = 4
    x_tar = (m1*x1 - m2*x2 + y2 - y1)/(m1-m2)
    y_tar = m2*(x_tar-x2)+y2
    p01k1 = np.array([[x_tar],[y_tar]])



    umax = np.array([[2], [2]])
    wmax = 1

    # ==========================================
    N=24
    # path1
    lambda1, path1 = create_position_path(p01, p01k1, N)

    # path2
    lambda2, path2 = create_position_path(p01k1, p01a, N)
    lambda2 += lambda1[-1]
    # path3
    lambda3, path3 = create_position_path(p01a, p01b, N)
    lambda3 += lambda2[-1]

    rot_path1 = create_orientation_path(r01, r01a, np.concatenate((lambda1, lambda2)))





    r01k1 = rot_path1[len(lambda1)]
    # path1
    T1, u1, w1 = calc_time(p01, p01k1, theta2(r01), theta2(r01k1), umax, wmax)
    #T1, u1 = calc_time(p01, p01k1, umax)
    # path2
    T2, u2, w2 = calc_time(p01k1, p01a, theta2(r01k1), theta2(r01a), umax, wmax)
    #T2, u2 = calc_time(p01k1, p01a, umax)
    # path3
    T3, u3, w3 = calc_time(p01a, p01b, theta2(r01a), theta2(r01b), umax, wmax)

    dt = 0.01
    
    #path1
    t1 = np.linspace(0,T1,int(T1/dt))
    NT1 = len(t1)
    pts1 = np.zeros((2, NT1))
    pts1[0,0] = p01[0][0]
    pts1[1,0] = p01[1][0]
    rots1 = np.zeros((NT1))
    rots1[0] = theta2(r01)
    for i in range(1,NT1):
        pts1[0,i] = pts1[0,i-1] + dt*u1[0][0]
        pts1[1,i] = pts1[1,i-1] + dt*u1[1][0]
        rots1[i] = rots1[i-1] + dt*w1
    

    # path2
    t2 = np.linspace(T1,T1+T2,int(T1+T2/dt))
    NT2 = len(t2)
    pts2 = np.zeros((2, NT2))
    pts2[0,0] = p01k1[0][0]
    pts2[1,0] = p01k1[1][0]
    rots2 = np.zeros((NT2))
    rots2[0] = theta2(r01k1)
    for i in range(1,NT2):
        pts2[0,i] = pts2[0,i-1] + dt*u2[0][0]
        pts2[1,i] = pts2[1,i-1] + dt*u2[1][0]
        rots2[i] = rots2[i-1] + dt*w2

    # path3
    t3 = np.linspace(T1+T2,T1+T2+T3,int(T1+T2+T3/dt))
    NT3 = len(t3)
    pts3 = np.zeros((2, NT3))
    pts3[0,0] = p01a[0][0]
    pts3[1,0] = p01a[1][0]
    rots3 = np.zeros((NT3))
    rots3[0] = theta2(r01a)
    for i in range(1,NT3):
        pts3[0,i] = pts3[0,i-1] + dt*u3[0][0]
        pts3[1,i] = pts3[1,i-1] + dt*u3[1][0]
        rots3[i] = rots3[i-1] + dt*w3


    # combined path

    t_complete = np.concatenate((t1, t2, t3))
    pts_complete = np.concatenate((np.transpose(pts1),
                                   np.transpose(pts2),
                                   np.transpose(pts3)))
    rots_complete = np.concatenate((rots1, rots2, rots3))

    # plotting
    fig, ax = plt.subplots(3)

    ax[0].plot(t_complete, pts_complete[:,0], 'r-')
    ax[1].plot(t_complete, pts_complete[:,1], 'r-')
    ax[2].plot(t_complete, rots_complete)

    ax[0].set_title('x-position vs time')
    ax[1].set_title('y-position vs time')
    ax[2].set_title('rotation vs. time')

    plt.setp(ax[0], ylabel='x-position (m)', xlabel='time (s)')
    plt.setp(ax[1], ylabel='y-position (m)', xlabel='time (s)')
    plt.setp(ax[2], ylabel='rotation (rad)', xlabel='time (s)')

    fig.tight_layout()
    fig.savefig('part3c1.png')
    plt.show()
    plt.close()

    print('Time to complete movement:',round(t_complete[-1],2),'s')

    # plotting (showing figure)
    display_width = 1000
    display_height = 1000
    display_scale = 200

    pygame.init()

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Project Part 3c')

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

    p01 *= display_scale
    p01k1 *= display_scale

    robot_list = []

    rots_complete = [rot2(each) for each in rots_complete]

    for i in range(len(pts_complete)):
        robot_list.append(circle_obj(r_radius, pts_complete[i,:]*display_scale, rots_complete[i], blue, 0, gameDisplay, display_height))

    person = circle_obj(p_radius, p02, r02, green, 0, gameDisplay, display_height)
    table = rect_obj(table_dim[0], table_dim[1], p03, r03, brown, gameDisplay, display_height)
    shelf = rect_obj(shelf_dim[0], shelf_dim[1], p04, r04, yellow, gameDisplay, display_height)

    path_points = [p01, p01k1, p01a, p01b]
    full_path = path_obj(path_points, green, gameDisplay, display_height)

    def draw_grid(num_blocks=2):
        for i in range(0,display_width,int(display_scale/num_blocks)):
            for j in range(0,display_height,int(display_scale/num_blocks)):
                rect = pygame.Rect(i, j, int(display_scale/num_blocks), int(display_scale/num_blocks))
                pygame.draw.rect(gameDisplay, white, rect, 1)
    

    crashed = False
    gameDisplay.fill(gray)
    draw_grid()
    i = 0
    while not crashed:
        gameDisplay.fill(gray)
        draw_grid()
        robot_list[i].draw()
        person.draw()
        table.draw()
        shelf.draw()
        full_path.show_path()

        if i >= len(robot_list)-1:
            pygame.display.update()
            time.sleep(1)
            crashed=True
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(gameDisplay, 'part3b.png') 
                crashed = True
            #print(event)

        pygame.display.update()
        if i == 0:
            time.sleep(1)
        time.sleep(dt)
        i += 1
    
    pygame.quit()




# ================================================================================= #
# =========================    Execute Project Sections    ======================== #
# ================================================================================= #


#run_part1()
#run_part2()
#run_part3b()
#run_part3c()
