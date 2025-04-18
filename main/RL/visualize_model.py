"""
visualize_model.py

This script loads a trained Proximal Policy Optimization (PPO) agent and uses it to interact with and render a custom environment defined in `env.py`.

The script creates an instance of the PPO agent, loads the saved model parameters, and runs the agent in the environment, rendering each step.

Usage:
    Run this script to visualize the performance of the trained PPO agent.

Example:
    $ python visualize_model.py
"""

from stable_baselines3 import PPO
from env import MyEnv  # Import your custom environment module
from params import *

env = MyEnv(
        agent_start_pos_longlat,
        goal_pos_longlat,
        heading,
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
        
        obstacle_motion_type,   
        random_goal_position_status=random_goal_position_status,
        
        simulation_status=True,
        record=True,
        video_name="simple_nav2",
    )
# check_env(env, warn=True)

# Create the environment
env.reset()
terminated = False
truncated = False   

# Load the trained model
model = PPO.load(r"RL\training\simple_navigation\best_model.zip")

# Reset the environment and get the initial observation
obs, _ = env.reset()

# Visualize the model
terminated = False
truncated = False
total_reward = 0
while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    total_reward += reward
print("Total Rewards: ", total_reward)
env.close()
