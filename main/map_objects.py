import pygame
from math import *
import numpy as np

import zmq, pickle
import logging

from RL.helper_funcs import longlat_to_xy
from utils.general import get_socket
from config import *

# establish socket
obj_process_sub = get_socket(PORTS['OBJ_PROCESS_PUB'], "SUB")

# map size in pixels
MAP_WIDTH = 600
MAP_HEIGHT = MAP_WIDTH

MARGINS = 20

CIRCLE_RADIUS = 10 

MAX_DIST = 10 # meters

MTR_2_PIX_RATIO = MAP_WIDTH / MAX_DIST  # pixel/meter
VEL_SCALE_RATIO = 3

MIN_VEL_LENGTH = 10

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 200, 0)
GREEN = (0, 255, 0)
PINK = (255, 100, 200)

# Grid settings
ROWS, COLS = 10, 10  # Number of rows and columns
CELL_WIDTH = MAP_WIDTH // COLS
CELL_HEIGHT = MAP_HEIGHT // ROWS


# set up the display
pygame.init()
window = pygame.display.set_mode((MAP_WIDTH+2*MARGINS, MAP_HEIGHT+2*MARGINS))
pygame.display.set_caption('Map')

# font 
font = pygame.font.Font(None, 20) 

def draw_text(text, pos, color=WHITE, background=BLACK):
    obj_text = font.render(text, True, color, background)
    window.blit(obj_text, pos)

def xy_to_pixel(xy: np.array):
    """Convert absolute xy coordinates of an object to the map xy pixel coordinates."""

    xy_pix = np.array([xy[0]*MTR_2_PIX_RATIO + MARGINS, 
              MAP_HEIGHT//2-xy[1]*MTR_2_PIX_RATIO + MARGINS]).astype(int)  
    
    return np.clip(xy_pix, 
                   [MARGINS, MARGINS], 
                   [MARGINS+MAP_WIDTH, MARGINS+MAP_WIDTH]) # limit coordinates to edges of grid

def draw_grid(color=BLACK):
    """Draws a grid on the screen."""
    for row in range(ROWS + 1):
        pygame.draw.line(window, color, (MARGINS, row * CELL_HEIGHT+MARGINS), (MAP_WIDTH+MARGINS, row * CELL_HEIGHT+MARGINS))
    for col in range(COLS + 1):
        pygame.draw.line(window, color, (col * CELL_WIDTH+MARGINS, MARGINS), (col * CELL_WIDTH+MARGINS, MAP_HEIGHT+MARGINS))

def draw_dot(coor, color, radius=CIRCLE_RADIUS):
    pygame.draw.circle(window, color, coor, radius)

def draw_map():
    window.fill(WHITE)
    draw_grid()

def draw_velocity(xy_pixel: np.ndarray, vel: np.ndarray, color=BLACK, thickness=3):
    """Draws the velocity vector of an object."""
    angle = np.arctan2(vel[1], vel[0])
    vel_pixel = vel*VEL_SCALE_RATIO*MTR_2_PIX_RATIO # get the length of the vector in terms of pixels
    vel_pixel = max(np.linalg.norm(vel_pixel), MIN_VEL_LENGTH)
    line_endpt = np.round(xy_pixel + vel_pixel*np.array([np.cos(angle),
                                                -np.sin(angle)
                                                ])) #np.array([vel_pixel[0], -vel_pixel[1]]).astype(int) # get endpoint of vector

    pygame.draw.line(window, color, xy_pixel, line_endpt, thickness)
    
def draw_obj(obj_info: dict):
    """Draw a detected obstacle on the map with its absolute position.

      
    """ 

    obj_xy = obj_info.xy_abs
    obj_xy_pix = xy_to_pixel(obj_xy)

    size = obj_info.size
    if 0 <= size <= 2:
        radius = 20
        color = BLUE
    elif size <= 4:
        radius = 30
        color = YELLOW
    else:
        radius = 40
        color = GREEN

    draw_dot(obj_xy_pix, color, radius=radius)

    obj_xy = np.round(obj_xy,2)
    # obj_info['velocity'] = np.round(obj_info['velocity'])
    draw_text(f"({obj_xy[0]},{obj_xy[1]})", obj_xy_pix+10)
    draw_text(f"({obj_info.id})",obj_xy_pix-10, BLACK, WHITE)
    # draw_text(f"{np.round(np.linalg.norm(obj_info.velocity_abs), 2)}", (obj_xy_pix[0]-20, obj_xy_pix[1]+10))
    # draw_velocity(xy_pixel=obj_xy_pix, vel=obj_info.velocity_abs, color=YELLOW)


def draw_boat(boat_info: dict):
    """Draw the representation of the boat on the map.
    
    Parameters
    ----------
    boat_info : dict
        Dictionary containing information about the boat. Expected keys:
            - "heading" : float
                Current heading of the boat in degrees.
            - "location" : numpy array of float
                Boat's current GPS location as [longitude, latitude] in degrees.
            - "velocity: : numpy array of float
                Boat's velocity vector as [vx, vy] in m/s
    """

    fc_xy = np.array(longlat_to_xy(boat_info["location"]))
    theta = radians(min((boat_info["heading"]-90)%360, 360-(boat_info["heading"]-90)%360))
    cam_xy = np.array([cos(theta), sin(theta)]) * CAMERA["elevation"] + fc_xy
    cam_xy = np.round(cam_xy, 2)

    draw_dot(xy_to_pixel(fc_xy), BLACK, 5) # draw flight controller position

    cam_xy_pix = xy_to_pixel(cam_xy)
    draw_dot(cam_xy_pix, RED) # draw camera position

    draw_text(f"({cam_xy[0]},{cam_xy[1]})", cam_xy_pix+10)

    draw_velocity(xy_pixel=cam_xy_pix, vel=boat_info["velocity"])

if __name__ == "__main__":

    running = True

    while running:
        
        
        processed_obs_list = pickle.loads(obj_process_sub.recv())
        # boat_info = nano_info[0]["sensors"]
        boat_info = {"location": [0,0],
                     "heading": 90,
                     "velocity": np.array([0,0])
                     }
        

        processed_obs_list = processed_obs_list

        draw_map()
        draw_boat(boat_info)

        for obj in processed_obs_list:
            draw_obj(obj_info=obj)            

        pygame.display.flip()  # Update the display
        
        for event in pygame.event.get(): # end program if user closes pygame window
            if event.type == pygame.QUIT:
                running = False

