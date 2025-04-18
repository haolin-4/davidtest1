"""
env.py

This module defines the MyEnv class, which represents the environment for the maritime simulation.
It includes the initialization of the environment, agent properties, and methods for resetting and stepping through the environment.
Maritime Agent Reinforcement Learning for Intelligent Navigation (MARLIN)

Classes:
    MyEnv: Custom environment for maritime simulation, inheriting from gym.Env.

Main Methods:
    __init__(self, etc):
        Initializes the environment with the given parameters.
    reset(self):
        Resets the environment to its initial state.
    step(self, action):
        Executes a step in the environment based on the given action.
    render(self):
        Renders the environment in pygame.

Main attributes:

    self.action_space: The action space for the environment. 
        Bounds:
            Normalised throttle output: [0, 1]
            Normalised nozzle angle: [-1,1]

    self.observation_space: The observation space for the environment. 
        Format:
        {'agent': [x_rel[-1,1], y_rel[-1,1], velocity[-1,1], heading[0,1], distance_to_goal[0, 1], angle_to_goal[0, 1]] 
        'obs1_active': [0,1], 'obs1_type': [0,1,2,3,4], 'obs1': [x_rel[-1,1], y_rel[-1,1], velocity[-1,1], heading[0,1], safety_radius], 
        'obs2_active': [0,1], 'obs2_type': [0,1,2,3,4], 'obs2': [x_rel[-1,1], y_rel[-1,1], velocity[-1,1], heading[0,1], safety_radius], 
        ...}
        
        where:
            obs_active:
                • 0 => inactive
                • 1 => active;
            obs_type:
                • 0 => unclassified
                • 1 => heading away
                • 2 => head on
                • 3 => crossing
                • 4 => overtaking
    
    self.state: The current state of the environment. Follows self.observation_space format.
    
    self.prev_state: The previous state of the environment. Mainly for rendering purposes. Follows self.observation_space format.

Usage:
    Import this module and create an instance of MyEnv.

Example:
    from env import MyEnv
"""

# Add file directory to system path and change working directory (to maintain imports)
import math
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
os.chdir(str(Path(__file__).parent.parent.resolve()))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import copy
import scipy.optimize as opt
import cv2
from RL.helper_funcs import *
from RL.reference_values import *

from collections import deque

class MyEnv(gym.Env):
    def __init__(
        self,
        agent_start_pos_longlat,
        goal_pos_longlat,
        heading_deg,
        max_velocity_knots,
        cruising_speed_knots,
        max_acc_ms2,
        max_yaw_rate_degs,
        detection_radius,
        min_obs_detection_radius,
        screen_height,
        screen_width,
        margins,
        ops_bubble_multiplier,
        grid_number,
        decision_rate,
        display_rate,
        colours_dict,
        max_obstacles,
        safety_radius_dict,
        rewards_weights_dict,
        entity_size,
        proximity_to_goal,
        
        ### DAVID's ADDITION START ###
        obstacle_motion_type,
        no_of_generated_obs=0,
        random_goal_position_status=True,
        
        simulation_status=True,
        record=False,
        video_name="Current",
        ### DAVID's ADDITION END ###
    ):
        super(MyEnv, self).__init__()
        
        
        # Agent properties
        self.max_velocity_ms = knotstoms(max_velocity_knots)
        self.cruising_speed_ms = knotstoms(cruising_speed_knots)
        self.max_acc_ms2 = max_acc_ms2
        self.max_yaw_rate_degs = max_yaw_rate_degs
        self.detection_radius = detection_radius

        # Navigational data
        self.goal_pos_xy = np.array(longlat_to_xy(goal_pos_longlat))
        self.goal_pos_xy_rel = np.array([0,0])                                      # goal pos is set as the origin 
        self.agent_start_pos_xy = np.array(longlat_to_xy(agent_start_pos_longlat))  
        self.agent_start_pos_xy_rel = self.agent_start_pos_xy - self.goal_pos_xy    # relative position to goal pos
        self.initial_heading_degs = heading_deg                     
        self.random_goal_pos_status = random_goal_position_status                 # determines if start position is random
        
        # Operation Environment for simulation
        self.ops_bubble_multiplier = ops_bubble_multiplier
        self.ops_COG, self.ops_bubble_radius, self.ops_bottom_left, self.ops_top_right, self.max_ops_dist = self.get_operational_environment()
        self.max_ops_dist_scalar = np.linalg.norm(self.max_ops_dist) # maximum distance to goal 
        self.decision_rate = decision_rate
        self.safety_radius_dict = safety_radius_dict 
        self.reward_weights_dict = rewards_weights_dict 
        self.proximity_to_goal = proximity_to_goal
        self.max_obstacles = max_obstacles
        self.max_no_of_generated_obs = no_of_generated_obs
        self.min_obs_velocity_ms = self.cruising_speed_ms * 0.3
        self.max_obs_velocity_ms = self.cruising_speed_ms * 1.0


        # Screen properties
        self.screen = None                              # initialized later in render()
        self.screen_height = screen_height              # pixels
        self.screen_width = screen_width                # pixels
        self.left_column_width = self.screen_height   # pixels
        self.margins = margins                          # pixels
        self.grid_number = grid_number + 1              # number of grids
        self.display_rate = display_rate
        self.steps = int(self.display_rate / self.decision_rate)
        self.size_pixels = max(self.metre_to_pixel(entity_size), 10) # size of all squares representing agent/obstacles/goal
        self.linewidth_pixels = self.left_column_width // 400  # width of squares
        self.grid_size = self.left_column_width // self.grid_number  # pixel size
        self.grid_scale = self.ops_bubble_radius * 2 / self.grid_number  # metres
        self.colours_dict = colours_dict
        # Find the closest scale (metres)
        self.closest_scale = min(
            [1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            key=lambda x: abs(x - self.grid_scale),
        )
        # Calculate the length of the line in pixels
        self.line_length_pixels = int(
            (self.closest_scale / (self.ops_bubble_radius * 2))
            * (self.left_column_width - 2 * self.margins)
        )
        
        # Define operating area on screen
        self.drawing_area = pygame.Rect(
            self.margins,
            self.margins,
            self.left_column_width - 2 * self.margins,
            self.left_column_width - 2 * self.margins,
        )  # left, top, width, height
        
        # Define normalized action space
        self.action_space = self.get_action_space()

        # Define observation space
        self.observation_space = self.get_observation_space()

        # Initialize states
        self.state = None
        self.prev_state = None

        self.time_step = 1 / self.decision_rate  # seconds
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maritime Environment")
        self.clock = pygame.time.Clock()

        # Set end time for truncation
        self.end_time = (
            4 * self.ops_bubble_radius / self.cruising_speed_ms 
        ) # seconds 
        self.elapsed_time = 0  # Initialize elapsed time

        ### SEAN's EDIT START ###
        # Collision boolean flag for each obstacle 
        self.collision_flags = [False] * self.max_obstacles
        ### SEAN's EDIT END ###
        
        ### David's EDIT START ###
        self.video_name = video_name
        self.record = record
        if self.record:         # Set up the video writer
            self.vid_holder = cv2.VideoWriter(f"{self.video_name}.mp4", 
                                           cv2.VideoWriter_fourcc(*'mp4v'),  # MP4 codec, 
                                           display_rate, 
                                           (self.screen_width, self.screen_height))
        
        # Dictionary for storing rewards
        self.rewards_log = self.prev_rewards_log = {} 
        
        # Predefined fonts for pygame screen
        self.large_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.medium_font = pygame.font.SysFont("segoeui", 14, bold=True)
        self.small_font = pygame.font.Font(None, 20)
             
        self.obstacle_motion_type = obstacle_motion_type
        self.simulation = simulation_status                 # Determines whether to be used for training or for real time visualisation
        
        # Track how long an obstacle is not active
        self.obs_dead_time_list = [0] * self.max_obstacles 
        # Maps environement obstacle id to tracker id from the YOLO object detection
        self.obs_to_tracker_id_dict = {}
        for i in range(1, self.max_obstacles+1): self.obs_to_tracker_id_dict[i] = -1
        
        self.agent_pixel_pos_deque = deque([])  # Track previous agent pos for drawing the trail line
        ### DAVID's EDIT END ###
        
    def reset(self, seed=None, options=None):

        ### DAVID's EDIT START ###
        # Normalise agent starting state 
        agent_start_pos_xy_norm = self.agent_start_pos_xy_rel / self.max_ops_dist
        
        start_pos_norm_list = []
        x_step = 0.1
        y_step = 0.1
        no_of_x_pts = 3
        no_of_y_pts = 7
        bottom_left = agent_start_pos_xy_norm - np.array([no_of_x_pts//2 * x_step,no_of_y_pts//2 * y_step])
        for i in range(no_of_x_pts):
            for n in range(no_of_y_pts):
                start_pos_norm_list.append(bottom_left+np.array([i*x_step, n*y_step]))
        
        if self.random_goal_pos_status:
            agent_start_pos_xy_norm = np.array([-0.8, -0.4]) #start_pos_norm_list[np.random.choice(len(start_pos_norm_list))]
         
        initial_speed_norm = self.cruising_speed_ms / self.max_velocity_ms
        initial_heading_norm = self.initial_heading_degs / 360.0
        initial_distance_to_goal_norm = np.linalg.norm(self.agent_start_pos_xy_rel) / self.max_ops_dist_scalar
        initial_angle_to_goal_norm = self.get_angle_to_goal(self.agent_start_pos_xy, 
                                                            self.goal_pos_xy, 
                                                            self.initial_heading_degs) / 180.0
        
        initial_heading_norm = np.random.choice([0, 45, 90, 135, 180, 225, 315]) / 360
            
        # Initialize the state dictionary     
        self.state = {
            "agent": np.append(
                agent_start_pos_xy_norm, # type: numpy array
                [initial_speed_norm, 
                 initial_heading_norm, 
                 initial_distance_to_goal_norm, 
                 initial_angle_to_goal_norm]
            ).astype(np.float64)
        }
        for i in range(1, self.max_obstacles + 1):
            self.state[f"obs{i}_active"] = 0  # Initially inactive
            self.state[f"obs{i}_type"] = 0  # unclassified
            self.state[f"obs{i}"] = np.append(self.state["agent"][:4], [0]).astype(np.float64)  # to keep it within bounds
        ### DAVID's EDIT END ###

        # Initialize previous state dictionary
        self.prev_state = copy.deepcopy(self.state)

        self.elapsed_time = 0  # Reset elapsed time
        
        ### SEAN's EDIT START ###
        self.collision_flags = [False] * self.max_obstacles
        ### SEAN's EDIT END ###       
        
        ### DAVID's EDIT START ###
        # Reset all rewards in rewards log to 0
        for key in self.rewards_log: self.rewards_log[key] = 0 
        ### DAVID's EDIT END ###
        
        return self.state, {}
    
    def step(self, action):                 
        self.prev_state = copy.deepcopy(self.state)  # before any changes
        
        # assign action elements
        normalized_acc, normalized_yaw_rate = action
        self.classify_obstacles()
        
        ### DAVID's EDIT START ###
        self.acc_ms2 = normalized_acc * self.max_acc_ms2 
        self.yaw_rate_degs = normalized_yaw_rate * self.max_yaw_rate_degs
        next_state = self.get_next_state(self.acc_ms2, self.yaw_rate_degs)  # update agent state
        ### DAVID's EDIT END ###
        
        # Check if agent is still in ops environment
        agent_xy = np.array(next_state[:2]*self.max_ops_dist+self.goal_pos_xy)
        in_ops_env = self.check_in_operational_environment(agent_xy)
        
        ### SEAN's EDIT START ###
        # Update agent's state
        self.state["agent"] = next_state
        ### SEAN's EDIT END ###

        goal_reached = bool(self.state['agent'][4]*self.max_ops_dist_scalar < self.proximity_to_goal)

        # Update, generate and classify obstacles
        self.update_obstacles()
        self.generate_obstacles()

        ### DAVID's EDIT START ###
        self.prev_rewards_log = self.rewards_log.copy() 
        reward = self.get_reward(normalized_acc, normalized_yaw_rate, in_ops_env, goal_reached)
        ### DAVID's EDIT END ###

        # Update elapsed time
        self.elapsed_time += self.time_step

        # Check if the episode is done
        terminated = goal_reached or not in_ops_env
        truncated = bool(self.elapsed_time >= self.end_time)  # Truncate if elapsed time exceeds end_time

        return self.state, reward, terminated, truncated, {}

    def external_update(self, sensor_data, processed_obs_list, new_goal_pos_longlat, action):
        """Update the environment observation space externally with boat and 
        obstacle data. Used in deployment."""
        
        self.prev_state = copy.deepcopy(self.state) # before any changes
        
        # assign action elements
        normalized_acc, normalized_yaw_rate = action
        
        self.acc_ms2 = normalized_acc * self.max_acc_ms2
        self.yaw_rate_degs = normalized_yaw_rate * self.max_yaw_rate_degs
        
        self.goal_pos_xy = np.array(longlat_to_xy(new_goal_pos_longlat))
        
        # Unpack agent state
        xy = np.array(longlat_to_xy([sensor_data.long, sensor_data.lat]))
        velocity_magnitude_ms = np.linalg.norm(np.array(sensor_data.velocity))
        heading = sensor_data.heading
        distance_to_goal = np.linalg.norm(xy - self.goal_pos_xy)
        angle_to_goal = self.get_angle_to_goal(xy, 
                                                    self.goal_pos_xy, 
                                                    self.initial_heading_degs) 
        
        # Update Ops env to fit the new goal pos and agent starting position
        if not self.check_in_operational_environment(xy):
            self.update_ops_env([sensor_data.long, sensor_data.lat], new_goal_pos_longlat)
            
        # Normalise agent state
        xy_norm = (xy - self.goal_pos_xy) / self.max_ops_dist
        velocity_norm = velocity_magnitude_ms / self.max_velocity_ms
        heading_norm = heading / 360
        distance_to_goal_norm = distance_to_goal / self.max_ops_dist_scalar
        angle_to_goal_norm = angle_to_goal / 180.0

        # Update agent state
        self.state["agent"] = np.append(xy_norm, [velocity_norm, 
                                                  heading_norm, 
                                                  distance_to_goal_norm, 
                                                  angle_to_goal_norm])
        
        # Process detected obstacles
        have_id = False
        for obs_data in processed_obs_list:
            
            tracker_id = obs_data.id
            
            if tracker_id not in self.obs_to_tracker_id_dict.values(): 
                # Check if any ids are available
                for i in range(1, self.max_no_of_generated_obs+1):
                    if self.obs_to_tracker_id_dict[i] == -1: 
                        self.obs_to_tracker_id_dict[i] = tracker_id
                        have_id = True
                        break
            
                if not have_id: continue # Skip the object if no ids availabele
            
            if math.nan in obs_data.xy_rel_raw: # Skip the object if there was 0 division
                continue
            
            # Get the corresponding obstacle id from the tracker id
            for i in self.obs_to_tracker_id_dict.keys():
                if self.obs_to_tracker_id_dict[i] == tracker_id:  
                    obs_id = i
            
            # Normalize values
            obs_xy_abs = obs_data.xy_abs
            obs_xy_norm = (obs_xy_abs - self.goal_pos_xy) / self.max_ops_dist
            
            obs_vel_abs_magnitude = np.linalg.norm(np.array(obs_data.velocity_abs))
            
            # Normalise obs state
            obs_velocity_norm = min(obs_vel_abs_magnitude / self.max_obs_velocity_ms, 1) # Limit the normalised obstacle velocity 
            obs_heading = np.rad2deg(np.arctan2(obs_data.velocity_abs[1], obs_data.velocity_abs[0]))
            obs_heading_norm = obs_heading / 360.0
            
            # Determine obs safety radius
            if obs_data.size <= 10: obs_size = "small"
            elif obs_data.size <= 20: obs_size = "medium"
            else: obs_size = "large"
            
            # Update obstacle state
            self.state[f"obs{obs_id}"]  = np.append(obs_xy_norm, [obs_velocity_norm, 
                                                                  obs_heading_norm, 
                                                                  self.safety_radius_dict[obs_size]])
            self.state[f'obs{obs_id}_active'] = 1 # Activate the obstacle
            
        for i in range(1, self.max_no_of_generated_obs+1):
            # Deactivate the obstacle if it is currently not detected
            if self.obs_to_tracker_id_dict[i] not in [obs_data.id for obs_data in processed_obs_list]:      
                self.state[f'obs{i}_active'] = 0 
            
            # Track how long the obstacle ID has been inactive
            if self.state[f'obs{i}_active'] == 0: self.obs_dead_time_list[i-1] += 1
                
            # Free the obstacle ID if it has not been active
            if self.obs_dead_time_list[i-1] > 10: self.obs_to_tracker_id_dict[i-1] = -1
        
        self.classify_obstacles()
        
        return self.state, distance_to_goal
        
    def update_ops_env(self, agent_start_pos_longlat, new_goal_pos_longlat):
        """Update the ops env with the new agent start pos and goal pos. Used in deployment"""
        
        self.goal_pos_xy = np.array(longlat_to_xy(new_goal_pos_longlat))
        self.agent_start_pos_xy = np.array(longlat_to_xy(agent_start_pos_longlat))
        self.agent_start_pos_xy_rel = self.agent_start_pos_xy - self.goal_pos_xy
        
        self.ops_COG, self.ops_bubble_radius, self.ops_bottom_left, self.ops_top_right, self.max_ops_dist = self.get_operational_environment()

        self.grid_scale = self.ops_bubble_radius * 2 / self.grid_number  # metres
        
        self.closest_scale = min(
            [1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            key=lambda x: abs(x - self.grid_scale),
        )
        
        # Set end time for truncation
        self.end_time = (
            4 * self.ops_bubble_radius / self.cruising_speed_ms 
        ) # seconds 

    def get_operational_environment(self):
        "Returns midpoint, ops_bubble_radius of operational environment"
        # midpoint coordinates
        midpoint = (self.agent_start_pos_xy+self.goal_pos_xy)/2
        
        # distance in metres between agent and goal
        distance = np.linalg.norm(self.agent_start_pos_xy_rel)

        # operational radius for training where the agent cannot exceed
        ops_bubble_radius = distance * self.ops_bubble_multiplier
        
        # Edges of the map
        min_xy = midpoint - ops_bubble_radius
        max_xy = midpoint + ops_bubble_radius
        
        max_x_dist = max(max_xy[0] - self.goal_pos_xy[0], 
                        self.goal_pos_xy[0] - min_xy[0])
        max_y_dist = max(max_xy[1] - self.goal_pos_xy[1], 
                              self.goal_pos_xy[1] - min_xy[1])
                
        max_ops_distance = np.array([max_x_dist, max_y_dist])
        
        return midpoint, ops_bubble_radius, min_xy, max_xy, max_ops_distance

    def check_in_operational_environment(self, pos_xy:np.ndarray):
        "Checks if a pos_xy point is in the operational environment"
        # distance of point from centre of ops environment
        dist_to_centre = np.linalg.norm(pos_xy - self.ops_COG)
        
        if dist_to_centre < self.ops_bubble_radius: 
            return True
        else: 
            return False

    def get_action_space(self):
        "Returns initialized action space."
        
        return spaces.Box(
            low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32
        )

    def get_observation_space(self):
        "Returns initialized observation space"

        # Initiali,ze the observation space dictionary
        observation_space_dict = {
            "agent": spaces.Box(
                low=np.array([-1, -1, 0, 0, 0, 0]),  # x_norm, y_norm, velocity_norm, heading_norm, distance_to_goal_norm, angle_to_goal_norm
                high=np.array([1,  1, 1, 1, 1, 1]),
                dtype=np.float64,
            )
        }

        # Loop to add obstacle spaces
        for i in range(1, self.max_obstacles + 1):
            # obs1_active
            observation_space_dict[f"obs{i}_active"] = spaces.Discrete(2)  # 0 if inactive, 1 if active
            # obs1_type
            observation_space_dict[f"obs{i}_type"] = spaces.Discrete(5)
            # 1 if heading away, 2 if head on, 3 if crossing, 4 for overtaking
            # obs1
            observation_space_dict[f"obs{i}"] = spaces.Box(
                low=np.array([-1, -1, 0, 0, 0]),
                high=np.array(
                    [1, 1, 1, 1, max(self.safety_radius_dict.values())]
                ),
                dtype=np.float64,
            )  # long, lat, velocity, heading, safety radius

        # Assign the observation space
        return spaces.Dict(observation_space_dict)

    def get_next_state(self, acc_ms2, yaw_rate_degs):
        "Returns self.state['agent'] after taking action"
        
        ### DAVID's EDIT START
        
        # Unpack agent state
        velocity = self.prev_state["agent"][2] * self.max_velocity_ms
        heading = self.prev_state["agent"][3] * 360
        xy_rel = self.prev_state["agent"][:2] * self.max_ops_dist
        
        # Get next position
        xy_rel += (velocity * self.time_step * 
            np.array(
                [np.cos(np.deg2rad(compass_to_math_angle(heading))), 
                 np.sin(np.deg2rad(compass_to_math_angle(heading)))]
                )
            )

        # Get next velocity
        velocity += acc_ms2 * self.time_step
        velocity = np.clip(velocity, 0, self.max_velocity_ms) 
        
        # Simulate drag to agent's linear motion *(not accurate representation of actual drag)
        if acc_ms2 == 0 and velocity > 0:
            drag_coefficient = 0.1
            wetted_area = 2 * 0.8 # metre ^2
            water_density = 1000 # kg/m^3
            mass = 40 # kg
            
            drag_force = 0.5 * water_density * drag_coefficient * wetted_area * velocity**2
            deceleration = drag_force / mass
            
            velocity = max(velocity - deceleration * self.time_step, 0)
            if velocity <= 0.1: velocity = 0
            
        heading += self.time_step * yaw_rate_degs
        heading = heading % 360  # Ensure heading is within the range [0, 360]
        
        # Normalise agent state
        xy_norm = xy_rel / self.max_ops_dist
        distance_to_goal_norm = np.linalg.norm(xy_rel) / self.max_ops_dist_scalar
        angle_to_goal_norm = self.get_angle_to_goal(xy_rel+self.goal_pos_xy, self.goal_pos_xy, heading) / 180.0

        return np.array(
            [xy_norm[0], 
             xy_norm[1], 
             velocity/self.max_velocity_ms, 
             heading/360.0,
             distance_to_goal_norm,
             angle_to_goal_norm], dtype=np.float64
        )
        ### DAVID's EDIT END

    def get_angle_to_goal(self, agent_xy, goal_pos_xy, agent_heading):
        """Get the angle difference between the agent's heading and the goal position"""
        
        # Get heading of goal relative to North from agent 
        goal_heading = heading_to_goal(xy_to_longlat(agent_xy), 
                                       xy_to_longlat(goal_pos_xy)) 
        angle_diff = (goal_heading - agent_heading) % 360 # Restrict angles to [0, 360]
        angle_diff = min(angle_diff, 360 - angle_diff) # Calculate the smallest angle difference between agent and goal heading

        return angle_diff

    def update_obstacles(self):
        """Checks if each obstacle is still in ops environment

        Either deactivates the obstacle or updates its position"""
        
        agent_xy_rel = self.state["agent"][:2] * self.max_ops_dist    
        
        for i in range(1, self.max_obstacles + 1):
            if self.state[f"obs{i}_active"] == 1:  # currently active obstacles
                
                ### DAVID's EDIT START ###
                # Unpack values
                obs_velocity = self.prev_state[f"obs{i}"][2] * self.max_obs_velocity_ms
                obs_heading = self.prev_state[f"obs{i}"][3] * 360
                obs_xy_rel = self.prev_state[f"obs{i}"][:2] * self.max_ops_dist
                
                # Get next obs position
                next_obs_xy_rel = obs_xy_rel + (obs_velocity * self.time_step *
                           np.array([np.cos(np.deg2rad(compass_to_math_angle(obs_heading))), 
                                     np.sin(np.deg2rad(compass_to_math_angle(obs_heading)))]))
                
                # Within detection radius
                if (np.linalg.norm(next_obs_xy_rel-agent_xy_rel) < self.detection_radius):  
                    self.state[f"obs{i}"][:2] = next_obs_xy_rel / self.max_ops_dist
                else:
                    self.state[f"obs{i}_active"] = 0  # deactivate obstacle

                ### DAVID's EDIT END ###
                
    def generate_obstacles(self):
        """Randomly decides whether to generate an obstacle and updates observation space"""
        
        ### DAVID's EDIT START ###
        # Unpack agent's current state
        agent_velocity = self.state["agent"][2] * self.max_velocity_ms
        agent_heading = self.state["agent"][3] * 360
        agent_xy_rel = self.state["agent"][:2] * self.max_ops_dist
        agent_dist_to_goal = self.state['agent'][4] * self.max_ops_dist_scalar
        ### DAVID's EDIT END ###
        
        for i in range(1, self.max_obstacles + 1):
            if self.state[f"obs{i}_active"] == 0:  # if obs is currently inactive
                generate = np.random.choice(
                    [True, False]
                    )  # randomly choose to generate or not
                
                if True:
                    
                    ### DAVID's EDIT START ###
                    obs_type = np.random.choice([1, 2, 3, 4])
                    
                    # randomly determine obs velocity based on obstacle type
                    obs_velocity = np.random.uniform(self.min_obs_velocity_ms, self.max_obs_velocity_ms)
                    
                    if self.obstacle_motion_type == 0: # Static obstacles
                        obs_velocity = 0
                    elif self.obstacle_motion_type == 3: # Mixed obstacles
                        if np.random.choice([0, 1]) == 0: # Randomly decide if they are static or constant motion
                            obs_velocity = 0

                    obs_size = np.random.choice(list(self.safety_radius_dict.keys())) # Randomly decide their size
                    obs_safety_radius = self.safety_radius_dict[obs_size]
                    
                    if obs_velocity == 0: # for static obstacles
                        obs_type = 0
                        
                        min_spawn_radius= self.detection_radius*0.5 
                        max_spawn_radius = self.detection_radius*0.8        # metres
                        spawn_radius = np.random.uniform(min_spawn_radius, max_spawn_radius)
                        
                        while True: # Randomly select obstacle position until it is within the ops env

                            rel_heading = random_sample((-100, -20), (20, 100)) # Relative angle of obs to agent heading
                            abs_heading = (agent_heading + rel_heading) % 360  # Ensure within 360 degrees    
                            # Calculate obstacle spawn position
                            obs_xy_rel = (agent_xy_rel + spawn_radius * 
                                            np.array(
                                                [np.cos(np.deg2rad(compass_to_math_angle(abs_heading))),
                                                np.sin(np.deg2rad(compass_to_math_angle(abs_heading)))]
                                            ) 
                            )
                            obs_xy_abs  = obs_xy_rel + self.goal_pos_xy

                            # Check if position is within the bounds of the ops area
                            if np.all(obs_xy_abs<=self.ops_top_right) and np.all(obs_xy_abs>=self.ops_bottom_left):
                                break
                            
                        obs_heading = np.random.uniform(0,360) # Pick a random heading for the obstacle (not impt since obs not moving)                            

                    else:   
                        # calculate collision point
                        if agent_velocity == 0:
                            time_to_collision = self.end_time - self.elapsed_time
                        else:
                            time_to_rch_goal = agent_dist_to_goal / agent_velocity
                            
                            time_upper_bound = time_to_rch_goal * 0.7
                            time_lower_bound = time_to_rch_goal * 0.4
                            
                            if time_upper_bound < 3.0:  # handle case where agent is too close to goal
                                time_to_collision = 3.0
                            else:
                                time_to_collision = np.random.uniform(
                                    time_lower_bound, time_upper_bound
                                )

                        
                        collision_xy = (agent_xy_rel + agent_velocity * time_to_collision * 
                                       np.array([
                                                np.cos(np.deg2rad(compass_to_math_angle(agent_heading))),
                                                np.sin(np.deg2rad(compass_to_math_angle(agent_heading)))
                                                 ])
                                       )

                        # type 1 will subsequently offset starting position to avoid obstacle
                        if obs_type == 1:  # heading away
                            rel_heading = random_sample((10.0, 170.0), (190.0, 350.0))
                            # relative heading of obstacle from collision point

                        elif obs_type == 2:  # head on
                            rel_heading = np.random.uniform(-10.0, 10.0)
                            # relative heading of obstacle from collision point

                        elif obs_type == 3:  # crossing
                            rel_heading = random_sample((40, 112.5), (247.5, 320))
                            # relative heading of obstacle from collision point

                        elif obs_type == 4:  # overtaking
                            rel_heading = np.random.uniform(170, 190)
                            # relative heading of obstacle from collision point
                            
                            obs_velocity = np.random.uniform(
                                self.max_obs_velocity_ms, agent_velocity* 0.3
                            )

                        # common to all obstacles
                        abs_heading = (agent_heading + rel_heading) % 360  # ensure within 360 degrees
                        
                        obs_distance = time_to_collision * obs_velocity  # from target point
                        
                        # calculate obstacle starting position
                        obs_xy_rel = (collision_xy + obs_distance * 
                                np.array(
                                    [np.cos(np.deg2rad(compass_to_math_angle(abs_heading))),
                                    np.sin(np.deg2rad(compass_to_math_angle(abs_heading)))]
                                    )
                                )

                        obs_heading = (
                            math_angle_to_compass(
                                np.degrees(np.arctan2(collision_xy[1] - obs_xy_rel[1], collision_xy[0] - obs_xy_rel[0]))
                            ) % 360
                        )
                        
                        if obs_type == 1:  # offset starting position to avoid obstacle
                            offset_heading = ( obs_heading + np.random.choice([90, -90])
                            ) % 360
                            obs_xy_rel += (obs_safety_radius* 5* 
                                np.array(
                                    [np.cos(np.deg2rad(compass_to_math_angle(offset_heading))),
                                    np.sin(np.deg2rad(compass_to_math_angle(offset_heading)))]
                                    )
                                )

                    obs_xy_norm = obs_xy_rel / self.max_ops_dist

                    # update obstacle information
                    self.state[f"obs{i}_active"] = 1  # Activate the obstacle
                    self.state[f"obs{i}"] = np.array(
                        [
                            obs_xy_norm[0],
                            obs_xy_norm[1],
                            obs_velocity/self.max_obs_velocity_ms,
                            obs_heading/360,
                            obs_safety_radius,
                        ],
                        dtype=np.float64,
                    )        
                    ### DAVID's EDIT END ###            
                    
    def classify_obstacles(self):
        "Classifies each obstacle based on its relative position to the agent, velocity and heading"
        
        ### DAVID's EDIT START ###
        # Unpack agent state
        agent_velocity = self.prev_state["agent"][2] * self.max_velocity_ms
        agent_heading = self.prev_state["agent"][3] * 360
        agent_xy = self.prev_state["agent"][:2] * self.max_ops_dist + self.goal_pos_xy
        
        for i in range(1, self.max_obstacles + 1):
            
            # Unpack obs state
            obs_velocity = self.prev_state[f"obs{i}"][2] * self.max_obs_velocity_ms
            obs_heading = self.prev_state[f"obs{i}"][3] * 360
            
            if self.state[f"obs{i}_active"] == 1:  # if active 
                closest_distance, t = self.closest_distance(
                    self.state["agent"], self.state[f"obs{i}"][:4]
                )

                # Calculate xy coordinates of closest distance to obs
                xy_closest_distance = (agent_xy + agent_velocity * t * 
                                       np.array(
                                           [np.cos(np.deg2rad(compass_to_math_angle(agent_heading))),
                                            np.sin(np.deg2rad(compass_to_math_angle(agent_heading)))]
                                            )
                )

                # difference in heading between agent and obstacle
                heading_diff = (obs_heading - agent_heading) % 360  # [0, 360]
                heading_diff = min(heading_diff, 360-heading_diff) 
                
                if obs_velocity < self.min_obs_velocity_ms: # Assume the obstacle is stationary if velocity too small
                    self.state[f"obs{i}_type"] = 0
                    self.state[f"obs{i}"][2] = 0
                elif closest_distance < self.state[f"obs{i}"][4] * 2 and self.check_in_operational_environment(xy_closest_distance):  
                    if 160 <= heading_diff < 200:
                        self.state[f"obs{i}_type"] = 2  # head on
                    elif 0 <= heading_diff < 20 or 340 <= heading_diff <= 360:
                        self.state[f"obs{i}_type"] = 4  # overtaking
                    else:
                        self.state[f"obs{i}_type"] = 3  # crossing

                else: # if point of closest distance is not in ops env, unlikely to collide
                    self.state[f"obs{i}_type"] = 1  # heading away
        ### DAVID's EDIT END ###

    def log_rewards(self, reward, reward_name):
        """Logs each reward/penalty to the rewards_log dict for display in logs table and
        analysis purposes. (Not an essential function)"""
        
        if reward_name == "total_reward": 
            if reward_name not in self.rewards_log: self.rewards_log[reward_name] = 0 # Initialise total_reward element
            self.rewards_log[reward_name] += reward
        else:
            self.rewards_log[reward_name] = reward 
        return 
    
    def get_reward(self, normalized_acc, normalized_yaw_rate, in_ops_env, goal_reached):
        "Calculates the total reward"
        
        ### DAVID's EDIT START ###
        agent_xy_rel = self.state["agent"][:2] * self.max_ops_dist
        
        prev_dist_to_goal = self.prev_state['agent'][4] * self.max_ops_dist_scalar
        dist_to_goal = self.state['agent'][4] * self.max_ops_dist_scalar
        
        prev_angle_to_goal = self.prev_state['agent'][5] * 180.0
        angle_to_goal = self.state['agent'][5] * 180.0

        # Reward moving towards the goal
        normalized_change_in_distance_to_goal = (prev_dist_to_goal-dist_to_goal) / (self.max_velocity_ms * self.time_step)
        distance_change_reward = normalized_change_in_distance_to_goal * self.reward_weights_dict["distance_change_penalty_weightage"]
        self.log_rewards(distance_change_reward, "distance_change_reward")
 
        # Penalize distance to goal
        normalized_proximity_to_goal = self.proximity_to_goal / (2 * self.ops_bubble_radius)
        distance_penalty = self.power_reward_func(
            (1, self.reward_weights_dict["distance_penalty_weightage"]), 
            (normalized_proximity_to_goal, 0), 
            max(dist_to_goal/self.max_ops_dist_scalar, normalized_proximity_to_goal),
            'up', 
            1
        )
        self.log_rewards(distance_penalty, "distance_penalty")
        
        # Penalty for not facing the goal
        angle_diff_penalty = self.power_reward_func(
            (1, self.reward_weights_dict["angle_diff_penalty_weightage"]),
            (0, 0),
            self.state['agent'][5],
            'up',
            1
        )
        self.log_rewards(angle_diff_penalty, "angle_diff_penalty")  
        
        # Reward turning towards the goal
        normalized_change_in_angle_diff = (prev_angle_to_goal - angle_to_goal) / (self.max_yaw_rate_degs*self.time_step)
        angle_change_reward = normalized_change_in_angle_diff * self.reward_weights_dict['angle_change_reward_weightage']
        self.log_rewards(angle_change_reward, "angle_change_reward")    
        
        # Time penalty (want agent to be efficient)
        time_penalty = self.reward_weights_dict["time_penalty_weightage"]
        self.log_rewards(time_penalty, "time_penalty")    

        # Penalize change in speed (acceleration)
        normalized_acc = abs(normalized_acc)    
        acc_penalty = normalized_acc * self.reward_weights_dict["acceleration_penalty_weightage"]
        self.log_rewards(acc_penalty, "acc_penalty")   
 
        # Penalize change in direction
        direction_penalty = (abs(normalized_yaw_rate) * self.reward_weights_dict["change_in_direction_penalty_weightage"])
        self.log_rewards(direction_penalty, "direction_penalty")   

        # Penalize exceeding operations environment
        exceed_ops_env_penalty = self.reward_weights_dict["exceed_ops_env_penalty_weightage"] if not in_ops_env else 0
        self.log_rewards(exceed_ops_env_penalty, "exceed_ops_env_penalty") 
        
        # Penalties to do with obstacles
        collision_penalty = sr_breach_penalty = 0
        for i in range(1, self.max_obstacles + 1):
            
            obs_xy_rel = np.array(self.state[f'obs{i}'][:2]) * self.max_ops_dist
            # print(f"{i} obs_rel_xy: {obs_xy_rel} agent_rel_xy {agent_xy_rel}")

            distance_from_obs = np.linalg.norm(agent_xy_rel-obs_xy_rel)
            safety_radius = self.state[f"obs{i}"][4] 

            if distance_from_obs <= 0.2*safety_radius:
                if self.collision_flags[i - 1] == False:
                    collision_penalty += self.reward_weights_dict["obs_collision_penalty_weightage"]
                    self.collision_flags[i - 1] = True
            else: 
                 self.collision_flags[i - 1] = False
            if distance_from_obs <= safety_radius:
                sr_breach_penalty += self.power_reward_func(
                    (1, 0), 
                    (0.2, self.reward_weights_dict["obs_SR_breach_penalty_weightage"]), 
                    max(distance_from_obs/safety_radius, 0.2),
                    'down', 
                    1
                )   
        self.log_rewards(collision_penalty, "collision_penalty") 
        self.log_rewards(collision_penalty, "sr_breach_penalty") 

        # Reward for reaching goal
        goal_reward = self.reward_weights_dict["goal_reward_weightage"] if goal_reached else 0
        self.log_rewards(goal_reward, "goal_reward") 

        
        
        # Penalty for breaching COLREGs
        # colregs_penalty = self.get_colregs_penalty()

        # Final reward
        total_reward = (
            distance_penalty
            + distance_change_reward
            + angle_diff_penalty
            + angle_change_reward
            + time_penalty
            + acc_penalty
            + direction_penalty
            + exceed_ops_env_penalty
            + goal_reward
            #+ obs_penalty
            + collision_penalty
            + sr_breach_penalty
        )
        self.log_rewards(total_reward, "total_reward")

        return total_reward
    
    def power_reward_func(self, pt1, pt2, cal_pt, concavity, power):
        if not (min(pt1[0], pt2[0]) <= cal_pt <= max(pt1[0], pt2[0])):
            raise ValueError("cal_pt must be between x1 and x2")
        
        if pt2[1] > pt1[1]: 
            big_pt = pt2
            small_pt = pt1
        else: 
            big_pt = pt1
            small_pt = pt2

        if concavity == 'down':
            ref_pt = big_pt
            final_pt = small_pt
        elif concavity == 'up':
            ref_pt = small_pt
            final_pt = big_pt
        else:
            raise TypeError('concavity is either up or down')

        x_2 = final_pt[0] - ref_pt[0]
        x = cal_pt - ref_pt[0]
        a_2 = final_pt[1] - ref_pt[1]

        if type(power) is int and power >= 1:
            reward = a_2 * (x/x_2)**power + ref_pt[1]
        else:
            raise TypeError('power must be an integer greater than 1')
        return reward
    
    def closest_distance(self, agent, obstacle):
        """Returns the closest projected distance between the 
        agent and an obstacle from the current time until end of episode

        agent = [x_norm, y_norm, velocity_norm, heading_norm];
        obstacle = [x_norm, y_norm, velocity_norm, heading_norm]"""

        # Unpack state
        agent_velocity = agent[2] * self.max_velocity_ms
        agent_heading = agent[3] * 360
        agent_xy_rel = np.array(agent[:2]) * self.max_ops_dist
        
        obs_velocity = obstacle[2] * self.max_velocity_ms
        obs_heading = obstacle[3] * 360
        obs_xy_rel = np.array(obstacle[:2]) * self.max_ops_dist

        agent_math_angle = compass_to_math_angle(agent_heading)
        obs_math_angle = compass_to_math_angle(obs_heading)

        def distance(t):
            new_agent_xy_rel = (agent_xy_rel + agent_velocity * t * 
                            np.array(
                                [np.cos(np.deg2rad(agent_math_angle)),
                                 np.sin(np.deg2rad(agent_math_angle))]
                            )
            )

            new_obs_xy_rel = (obs_xy_rel + obs_velocity * t * 
                        np.array(
                                [np.cos(np.deg2rad(obs_math_angle)),
                                 np.sin(np.deg2rad(obs_math_angle))]
                            )
            )

            return np.linalg.norm(new_agent_xy_rel - new_obs_xy_rel)

        result = opt.minimize_scalar(
            distance,
            bounds=(0, self.end_time - self.elapsed_time),
            method="bounded",
        )

        return result.fun, result.x     # time_at_closest_distance, closest_distance

    def xy_to_pixel(self, xy):
        "Converts xy to pixel coordinates"

        # Scale the coordinates to fit within the screen dimensions
        pixel_x = int(
            self.margins
            + ((xy[0] - self.ops_bottom_left[0]) / (2 * self.ops_bubble_radius))
            * (self.left_column_width - 2 * self.margins)
        )
        pixel_y = int(
            self.margins
            + ((self.ops_top_right[1] - xy[1]) / (2 * self.ops_bubble_radius))
            * (self.left_column_width - 2 * self.margins)
        )

        return np.array([pixel_x, pixel_y])

    def metre_to_pixel(self, metres):
        "Converts metres to number of pixels"
        return int(
            metres
            / (2 * self.ops_bubble_radius)
            * (self.left_column_width - 2 * self.margins)
        )

    def render(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        ### DAVID's EDIT START ###
        if self.simulation: # interpolate positions of entites during simulated lag

            for j in range(self.steps):
                
                # Draw the dark blue background in the drawing area
                pygame.draw.rect(self.screen, DARK_BLUE, self.drawing_area)        
                self.draw_grid()
                self.draw_goal()
                
                # Drawing (dynamic agent)
                self.draw_agent(interpolate=True, j=j)
                
                # Drawing (dynamic obstacles)
                self.draw_obstacles(interpolate=True, j=j)
                
                # Draw static elements
                self.draw_margins()
                self.draw_game_status()
                self.draw_north_arrow()
                self.draw_display_scale()
                self.draw_obs_types()
                pygame.draw.circle(
                    self.screen, 
                    LIGHT_GREY, 
                    self.xy_to_pixel(self.ops_COG), 
                    self.metre_to_pixel(self.ops_bubble_radius), 
                    width=self.linewidth_pixels
                ) # Draw ops bubble radius circle
                
                self.draw_agent_properties()
                self.draw_reward_logs_table()
                self.draw_wasd()
                
                # Update screen
                pygame.display.flip() 
                
                # Capture the screen
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if self.record: self.vid_holder.write(frame)
                
                self.clock.tick(self.display_rate)
                
                # Return pygame screen as an image 
                return frame

        else: # show live positions of entities
            
           # Draw the dark blue background in the drawing area
            pygame.draw.rect(self.screen, DARK_BLUE, self.drawing_area)        
            self.draw_grid()
            self.draw_goal()
            
            # *** Agent and obstacles are plotted on their exact pos, no interpolation 
            # Drawing (dynamic agent)
            self.draw_agent()
            # Drawing (dynamic obstacles)
            self.draw_obstacles()
            
            # Draw static elements
            self.draw_margins()
            self.draw_game_status()
            self.draw_north_arrow()
            self.draw_display_scale()
            self.draw_obs_types()
            pygame.draw.circle(
                self.screen, 
                LIGHT_GREY, 
                self.xy_to_pixel(self.ops_COG), 
                self.metre_to_pixel(self.ops_bubble_radius), 
                width=self.linewidth_pixels
            ) # Draw ops bubble radius circle
            
            self.draw_agent_properties()
            # self.draw_reward_logs_table()

            # Update screen
            pygame.display.flip() 
            
            # Capture the screen
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.record: self.vid_holder.write(frame)
            
            self.clock.tick(self.display_rate)
            return frame

    def draw_agent(self, interpolate=False, j=None):
        "Draws agent"
        
        ### DAVID's EDIT START
        
        # Unpack agent state
        agent_velocity = self.prev_state["agent"][2] * self.max_velocity_ms
        agent_heading = self.prev_state["agent"][3] * 360
        agent_xy = self.prev_state["agent"][:2] * self.max_ops_dist + self.goal_pos_xy
        next_agent_xy = self.state["agent"][:2] * self.max_ops_dist + self.goal_pos_xy
        
        if interpolate:
            interpolated_agent_pos_arr = self.interpolate_pixel_pos(
                self.xy_to_pixel(agent_xy),
                self.xy_to_pixel(next_agent_xy),
            )

            pixel_xy = interpolated_agent_pos_arr[j]
            
        else:
            pixel_xy = self.xy_to_pixel(agent_xy)      
            
        # Track the agent's position for drawing the beeline
        self.agent_pixel_pos_deque.append(pixel_xy)
        if len(self.agent_pixel_pos_deque) > 100: self.agent_pixel_pos_deque.popleft() 
        
        rect = pygame.Rect(
            pixel_xy[0] - self.size_pixels // 2,
            pixel_xy[1] - self.size_pixels // 2,
            self.size_pixels,
            self.size_pixels,
        )
        # Draw the square
        pygame.draw.rect(self.screen, YELLOW, rect, width=self.linewidth_pixels)

        arrow_length = max(
            min(
                self.metre_to_pixel(agent_velocity) * VELOCITY_ARROW_SCALE, 
                MAX_ARROW_LENGTH_PIXELS
            ),  
            MIN_ARROW_LENGTH_PIXELS)
        
        # Calculate the end point of the arrow
        end_xy = (pixel_xy + arrow_length * 
                 np.array(
                     [np.cos(np.deg2rad(compass_to_math_angle(agent_heading))),
                      -np.sin(np.deg2rad(compass_to_math_angle(agent_heading)))])
        ).astype(int)
        # Draw arrow
        pygame.draw.line(
            self.screen,
            YELLOW,
            pixel_xy,
            end_xy,
            width=self.linewidth_pixels,
        )
        # Detection radius circle
        pygame.draw.circle(
            self.screen,
            YELLOW, 
            pixel_xy,
            self.metre_to_pixel(self.detection_radius),
            width=self.linewidth_pixels,
        )

        # # Draw a beeline behind the agent to visualise its motion
        # for i, pos in enumerate(list(self.agent_pixel_pos_deque)[::-1]):
        #     if i % 5 == 0:
        #         pygame.draw.circle(
        #             self.screen, LIGHT_GREY, pos, self.linewidth_pixels,
        #         )

    def draw_obstacles(self, interpolate=False, j=None):
        """Draws all obstacles. 
        Set interpolate=True when simulating lag. 
        j: display_rate step"""
        
        ### DAVID's EDIT START
        for i in range(1, self.max_obstacles + 1):  # loop through obstacles
            
            if (self.state[f"obs{i}_active"] == 1):  # active
                
                # Unpack obs state
                safety_radius = self.prev_state[f"obs{i}"][4]
                obs_type = self.prev_state[f"obs{i}_type"]
                
                obs_velocity = self.prev_state[f"obs{i}"][2] * self.max_obs_velocity_ms
                obs_heading = self.prev_state[f"obs{i}"][3] * 360
                obs_xy = self.prev_state[f"obs{i}"][:2] * self.max_ops_dist + self.goal_pos_xy
                next_obs_xy = self.state[f"obs{i}"][:2] * self.max_ops_dist + self.goal_pos_xy
                
                if interpolate: # Interpolate obs position
                    
                    if self.prev_state[f"obs{i}_active"] == 1: # Draw obstacle if it is active
                        interpolated_obs_pos_arr = self.interpolate_pixel_pos(
                            self.xy_to_pixel(obs_xy),
                            self.xy_to_pixel(next_obs_xy)
                        )
                        pixel_xy = interpolated_obs_pos_arr[j]
                    else:
                        continue
                else:
                    pixel_xy = self.xy_to_pixel(obs_xy)

                # print(f"{i} pixel_xy: {pixel_xy}")
                if self.drawing_area.collidepoint(pixel_xy): # Check if the xy position is within the map_area (pygame.rect)

                    colour = self.colours_dict[obs_type][1]
                    obs_square = pygame.Rect(
                        pixel_xy[0] - self.size_pixels // 2,
                        pixel_xy[1] - self.size_pixels // 2,
                        self.size_pixels,
                        self.size_pixels,
                    )
                    # Draw the square
                    pygame.draw.rect(
                        self.screen, colour, obs_square, width=self.linewidth_pixels
                    )

                    ### DAVID's EDIT START ####
                    # Arrow length is proportional to velocity of object (in pixels)
                    arrow_length = max(
                        min(
                            self.metre_to_pixel(obs_velocity) * VELOCITY_ARROW_SCALE, 
                            MAX_ARROW_LENGTH_PIXELS
                        ), 
                        MIN_ARROW_LENGTH_PIXELS) 

                    # Calculate the end point of the arrow
                    end_xy = (pixel_xy + arrow_length * 
                              np.array(
                                  [np.cos(np.deg2rad(compass_to_math_angle(obs_heading))),
                                   -np.sin(np.deg2rad(compass_to_math_angle(obs_heading)))])
                    ).astype(int)
                    ### DAVID's EDIT END ####
                
                    # Draw arrow
                    pygame.draw.line(
                        self.screen,
                        colour,
                        pixel_xy,
                        end_xy,
                        width=self.linewidth_pixels,
                    )

                    SR_pixels = self.metre_to_pixel(safety_radius)

                    # Draw the safety radius circle 
                    pygame.draw.circle(
                        self.screen,
                        colour,
                        pixel_xy,
                        SR_pixels,
                        width=self.linewidth_pixels,
                    )

                    ### DAVID's EDIT START
                    # Draw the obstacle id 
                    font = pygame.font.SysFont(None, 20) 
                    obj_id_text = font.render(str(i), 
                                            True, 
                                            colour)
                    text_w, text_h = obj_id_text.get_size()
                    self.screen.blit(obj_id_text, (pixel_xy[0]+self.size_pixels//2+5, pixel_xy[1]-text_h//2-1))
                    ### DAVID's EDIT END

    def draw_goal(self):
        "Draws goal"
        pixel_xy = self.xy_to_pixel(self.goal_pos_xy)
        pygame.draw.circle(
            self.screen, GREEN, pixel_xy, self.size_pixels // 2
        )  # Green goal
    
        ### DAVID's ADDITION START
        pygame.draw.circle(
            self.screen, GREEN, pixel_xy, self.metre_to_pixel(self.proximity_to_goal), width=self.linewidth_pixels
        )  # Proximity circle
        ### DAVID's ADDITION END ###
        
    def draw_grid(self):
        "Draw gridlines"
        # Draw vertical grid lines
        for x in range(
            self.drawing_area.left,
            self.drawing_area.right + self.grid_size,
            self.grid_size,
        ):
            pygame.draw.line(
                self.screen,
                LIGHT_GREY,
                (x, self.drawing_area.top),
                (x, self.drawing_area.bottom),
            )
        # Draw horizontal grid lines
        for y in range(
            self.drawing_area.top,
            self.drawing_area.bottom + self.grid_size,
            self.grid_size,
        ):
            pygame.draw.line(
                self.screen,
                LIGHT_GREY,
                (self.drawing_area.left, y),
                (self.drawing_area.right, y),
            )

    def draw_display_scale(self):
        "Draws the display scale in metres at bottom right corner"

        x = self.left_column_width - self.margins
        y = self.left_column_width - (4 * self.margins // 5)

        # Draw the scale
        text_surface = self.medium_font.render(f"{self.closest_scale}m", True, WHITE)
        w, h = text_surface.get_size()
        x -= w
        self.screen.blit(text_surface, (x, y))  

        # Draw the horizontal line above the scale text
        line_end_x = x - 10  # Position the line next to the text
        line_start_x = line_end_x - self.line_length_pixels
        line_y = y+h/2
        pygame.draw.line(
            self.screen,
            WHITE,
            (line_start_x, line_y),
            (line_end_x, line_y),
            width=self.left_column_width // 200,
        )

    ### DAVID's ADDITION START
    def draw_game_status(self):
        """Displays the collision status of the agent with obstacles"""
        
        x, y = self.margins, self.left_column_width // 100
        word_spacing = 10
        
        collision_status = "Safe"
        collided_obs_text = ""
        color = GREEN
        for i in range(self.max_obstacles):
            if self.collision_flags[i] == True: #  check if collided with any obstacles
                collision_status = "Collided" 
                collided_obs_text + f"{i} "
                color = RED
                
        w, h = self.draw_text("Collision Status: ", (x, y),self.large_font) # Position the text at the top-left corner
        x += w
        w, h = self.draw_text(collision_status, (x, y),self.large_font, color)
        x += w+word_spacing
        if collision_status == "Collided":
            w, h = self.draw_text(f"Obstacles: {collided_obs_text}", (x, y),self.large_font)
            x += w
        self.draw_text(f"Time Elapsed: {self.elapsed_time:.1f}/{self.end_time:.1f}s Distance to Goal: {(self.state['agent'][4]*self.max_ops_dist_scalar):.0f} m", 
                (x, y),self.large_font)
    ### DAVID's ADDITION END

    def draw_north_arrow(self):
        "Draws north arrow"
        arrow_size = self.left_column_width // 50
        arrow_pos = (self.left_column_width - 1.5 * arrow_size, 2 * arrow_size)
        arrow_vertices = [
            (arrow_pos[0], arrow_pos[1] - arrow_size),  # Top
            (
                arrow_pos[0] - arrow_size // 2,
                arrow_pos[1] + arrow_size // 2,
            ),  # Bottom left
            (arrow_pos[0], arrow_pos[1] - arrow_size // 6),  # bottom kink
            (
                arrow_pos[0] + arrow_size // 2,
                arrow_pos[1] + arrow_size // 2,
            ),  # Bottom right
        ]
        pygame.draw.polygon(self.screen, WHITE, arrow_vertices)

        # Label the arrow with "N"
        self.draw_text("N",             
            (
                arrow_pos[0] - 7,
                arrow_pos[1] + 14,
            ), # Position below the arrow
            self.large_font) 

    def draw_margins(self):
        "Draws black margins (Covers parts of the obs/agent that is outside the map)"
        ### DAVID's EDIT START ###
        pygame.draw.rect(
            self.screen, BLACK, (0, 0, self.screen_width, self.margins)
        )  # Top margin
        pygame.draw.rect(
            self.screen, BLACK, (0, 0, self.margins, self.screen_height)
        )  # Left margin
        pygame.draw.rect(
            self.screen,
            BLACK,
            (0, self.screen_height - self.margins + 1, self.screen_width, self.margins),
        )  # Bottom margin
        pygame.draw.rect(
            self.screen,
            BLACK,
            (self.left_column_width - self.margins, 0, self.screen_width-(self.left_column_width-self.margins)
             , self.screen_height),
        )  # Right margin
        
        ### DAVID's EDIT END ###

    def draw_obs_types(self):
        "Displays colour-coded text for obstacle types"

        x = self.margins  # Initial x position for text
        y = self.left_column_width - (self.margins//3 * 2)  # y position for text

        for _, value in self.colours_dict.items():
            # Position the text at the bottom left corner
            w, h = self.draw_text(value[0], (x,y),self.medium_font, value[1])

            # Update x position for the next piece of text
            x += w + 10  # Add some space between texts

    def draw_agent_properties(self):
        "Display agent's state information"

        ### DAVID's EDIT START ###
        agent_velocity_knots = mstoknots(self.prev_state["agent"][2] * self.max_velocity_ms)
        agent_heading = self.prev_state["agent"][3] * 360
        agent_xy = np.array(self.prev_state["agent"][:2]) * self.max_ops_dist + self.goal_pos_xy
        agent_longlat = xy_to_longlat(agent_xy)

        x = self.left_column_width
        y = self.screen_height//8
        row_spacing = 10
        
        w, h = self.draw_text(f"Longitude: {agent_longlat[0]:.6f}°", 
                       (x, y),self.large_font)
        y += row_spacing+h
        w, h = self.draw_text(f"Latitude: {agent_longlat[1]:.6f}°", 
                       (x, y),self.large_font)   
        y += row_spacing+h
        w, h = self.draw_text(f"Velocity: {agent_velocity_knots:.1f} knots", 
                       (x, y),self.large_font)   
        y += row_spacing+h
        w, h = self.draw_text(f"Heading: {agent_heading:.0f}°", 
                       (x, y),self.large_font)  
        y += row_spacing+h
        w, h = self.draw_text(f"Acceleration: {self.acc_ms2:.4f}m/s^2", 
                       (x, y),self.large_font)
        y += row_spacing+h 
        w, h = self.draw_text(f"Yaw Rate: {self.yaw_rate_degs:.2f}°/s", 
                       (x, y),self.large_font)  
        ### DAVID's EDIT END ###
    
    ### DAVID's ADDITION START ###
    # Draw an arrow to represent increase or decrease in reward value
    def draw_change_arrow(self, x, y, direction, color): 
        y+=1
        if direction == "up":
            points = [(x, y), (x + 5, y+10), (x - 5, y+10)]
        elif direction == "down":
            points = [(x, y+10), (x + 5, y), (x - 5, y)]
        pygame.draw.polygon(self.screen, color, points)

    # Render the rewards table
    def draw_reward_logs_table(self):
        
        start_pos=(self.left_column_width, 3*self.screen_height//8)
        x, y = start_pos
        row_spacing=20
        value_x = x + 200
        
        # Draw the table header
        self.draw_text("Rewards", (x,y), self.large_font)
        pygame.draw.line(self.screen, WHITE, (x, y+30), (self.left_column_width+300-50, y+30), 2)

        y += 40

        # Render each reward
        for reward_name, value in self.rewards_log.items():
            if reward_name == "total_reward":  # Skip to render total reward at the end
                continue

            # Draw reward name
            self.draw_text(reward_name, (x,y), self.small_font, LIGHT_GREY)
            w, h = self.draw_text(round(value, 2), (value_x,y), self.small_font, WHITE)
            arrow_x = value_x+w+10

            # Check if value has changed and draw an arrow
            if reward_name in self.prev_rewards_log:
                if value > self.prev_rewards_log[reward_name]:
                    self.draw_change_arrow(arrow_x, y, "up", GREEN)  # Green for increase
                elif value < self.prev_rewards_log[reward_name]:
                    self.draw_change_arrow(arrow_x, y, "down", RED)  # Red for decrease

            y += row_spacing

        if "total_reward" in self.rewards_log:
            # Render total self.rewards_log at the bottom
            total_reward_value = self.rewards_log["total_reward"]

            # Draw total rewards
            self.draw_text("Total Reward", (x,y), self.small_font, YELLOW)
            w, h = self.draw_text(round(total_reward_value, 2), (value_x, y), self.small_font, YELLOW)
            arrow_x = value_x + w + 10

            # Check if total rewards changed and draw an arrow
            if "total_reward" in self.prev_rewards_log:
                if total_reward_value > self.prev_rewards_log["total_reward"]:
                    self.draw_change_arrow(arrow_x, y, "up", GREEN)  # Green for increase
                elif total_reward_value < self.prev_rewards_log["total_reward"]:
                    self.draw_change_arrow(arrow_x, y, "down", RED)  # Red for decrease

    def draw_text(self, text, xy, font, color=WHITE, background=None):
        """Draws text on the screen. Returns the width and height of the text. (w,h)"""
        text_surface = font.render(str(text), True, color, background)
        self.screen.blit(text_surface, xy)  
        return text_surface.get_size()
    
    def draw_wasd(self):
        """Draw WASD keys onto map"""

        key_size = 40
        space = 10
        # Top left position of W key
        x1, y1 = (self.left_column_width-self.margins-2*space-2*key_size, self.screen_height-self.margins-2*space-2*key_size)
        
        # Define key positions
        keys = {
            "W": (x1, y1),
            "A": (x1-space-key_size, y1+space+key_size),
            " ": (x1, y1+space+key_size),
            "D": (x1+space+key_size, y1+space+key_size),
        }
                
        for key, (x, y) in keys.items():
            pygame.draw.rect(self.screen, BLACK, (x, y, key_size, key_size), border_radius=10)
            self.draw_text(key, (x+15,y+10), self.medium_font, WHITE)
    ### DAVID's ADDITION END ###
       
    def interpolate_pixel_pos(self, pixels_start, pixels_end):
        "Interpolates between (pixel_x_start, pixel_y_start) and pixels_end = (pixel_x_end, pixel_y_end), returns array of tuples"
        x = np.linspace(pixels_start[0], pixels_end[0], self.steps)
        y = np.linspace(pixels_start[1], pixels_end[1], self.steps)
        return [(int(x[i]), int(y[i])) for i in range(self.steps)]

    def close(self):
        # self.out.release()
        pygame.quit()
        cv2.destroyAllWindows()
        
# end of class
