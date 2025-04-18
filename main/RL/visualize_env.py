"""
visualize_env.py

This module is responsible for visualizing the maritime simulation.
It sets up the environment, runs the simulation loop, and renders the environment.

The script initializes the custom environment `MyEnv`. It then runs the simulation loop, rendering the environment at each step.

Usage:
    Run this script to visualize the maritime simulation environment. Press Esc key to exit the simulation. If uninterrupted simulation will
    run until self.end_time is reached or agent has reached the goal. 
    Use the W, A & D keys to control the agent to test the simulation environment.

Example:
    $ python visualize_env.py
"""
from env import MyEnv
from params import *
import pygame
    
if __name__ == "__main__":

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
        video_name="env_demo",
    )
    # check_env(env, warn=True)

    # Create the environment
    env.reset()
    terminated = False
    truncated = False
    
    total_reward = 0
    
    while not (terminated or truncated):
        
        # Control the agent's position using W, A & D keys
        for event in pygame.event.get():
            
            # Check keys continuously
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_s]:  
                if keys[pygame.K_w]: # Increase acceleration
                    acceleration_normalized = 1
            else:
                acceleration_normalized = 0
            if keys[pygame.K_a] or keys[pygame.K_d]: 
                if keys[pygame.K_a]:  # Increase yaw to the left
                    yaw_rate_normalized = -1
                if keys[pygame.K_d]:  # Increase yaw to the right
                    yaw_rate_normalized = 1
            else:
                yaw_rate_normalized = 0

        acceleration_normalized = max(0, min(acceleration_normalized, 1))
        yaw_rate_normalized = max(-1, min(yaw_rate_normalized, 1))
        
        action = (acceleration_normalized, yaw_rate_normalized)
        # action = env.action_space.sample()  # Sample random action

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        frame = env.render()  # Render the environment

    print("Total Rewards: ", total_reward)

    env.close()

