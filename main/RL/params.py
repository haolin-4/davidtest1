"""
params.py

This module defines the parameters for the maritime simulation: environment, agent, display, and penalties.
"""
# Add file directory to system path and change working directory (to maintain imports)
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
os.chdir(str(Path(__file__).parent.parent.resolve()))

from RL.reference_values import *
from RL.helper_funcs import *

# Display (square screen)
screen_height = 800
right_column_width = 300
screen_width = screen_height + right_column_width
margins = 48
grid_number = 8
display_rate = 30  # Hz
entity_size = 8  # of agent, obstacles, goal (metres)
colours_dict = {
    0: ["Unclassified", WHITE],
    1: ["Heading Away", LIME],
    2: ["Head-on", RED],
    3: ["Crossing", CYAN],
    4: ["Overtaking", ORANGE],
}

# Environment
agent_start_pos_longlat = (103.859500,1.4221174)
goal_pos_longlat = (103.86100,1.4221174) 
heading = 90.0  # 90 degrees = east
decision_rate = 30  # Hz
ops_bubble_multiplier = 0.75
proximity_to_goal = 5  # metres
max_obstacles = 10
no_of_generated_obs = 5 
min_obs_detection_radius = 50  # metres (Unused)
obstacle_motion_type = 0 # [0:static, 1:constant speed, 2:dynamic, 3:mixed]
simulation_status = False
random_goal_position_status = True # yet to be implemented (lack of time)

# Penalty weightages
distance_penalty_weightage = 0
distance_change_reward_weightage = 1  
angle_diff_penalty_weightage = 0
angle_change_reward = 1
time_penalty_weightage = -1 
acceleration_penalty_weightage = 0 
change_in_direction_penalty_weightage = 0 
exceed_ops_env_penalty_weightage = -3000 
goal_reward_weightage = 2000
obs_collision_penalty_weightage = -100
obs_SR_breach_penalty_weightage = 0 # -50
# COLREGs (Not Yet Used)
obs_head_on_penalty_weightage = -5  # turn to starboard
obs_crossing_starboard_penalty_weightage = 0  # turn to starboard
obs_crossing_port_penalty_weightage = -1  # do not turn left
obs_crossing_port_direction_change_penalty_weightage = -1  # continue on current course
obs_overtaking_penalty_weightage = -1  # do not cross in front of obstacle

rewards_weights_dict = {
    "distance_penalty_weightage": distance_penalty_weightage,
    "distance_change_penalty_weightage": distance_change_reward_weightage,
    "angle_diff_penalty_weightage": angle_diff_penalty_weightage,
    "angle_change_reward_weightage": angle_change_reward, 
    "time_penalty_weightage": time_penalty_weightage,
    "acceleration_penalty_weightage": acceleration_penalty_weightage,
    "change_in_direction_penalty_weightage": change_in_direction_penalty_weightage,
    "exceed_ops_env_penalty_weightage": exceed_ops_env_penalty_weightage,
    "goal_reward_weightage": goal_reward_weightage,
    "obs_collision_penalty_weightage": obs_collision_penalty_weightage,
    "obs_SR_breach_penalty_weightage": obs_SR_breach_penalty_weightage,
    "obs_head_on_penalty_weightage": obs_head_on_penalty_weightage,
    "obs_crossing_starboard_penalty_weightage": obs_crossing_starboard_penalty_weightage,
    "obs_crossing_port_penalty_weightage": obs_crossing_port_penalty_weightage,
    "obs_overtaking_penalty_weightage": obs_overtaking_penalty_weightage,
}

# Agent properties
max_velocity_knots = 33
cruising_speed_knots = max_velocity_knots / 3
max_acc_ms2 = 2  # meters per second squared
max_yaw_rate_degs = 45  # per second
detection_radius = 100  # metres

# Safety Radius (metres)
small_radius = 30
medium_radius = 40
large_radius = 60

safety_radius_dict = {
    "small": small_radius,
    "medium": medium_radius,
    "large": large_radius,
}
