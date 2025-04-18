"""
train.py

This script trains a Proximal Policy Optimization (PPO) agent using a custom environment defined in `env.py`.

The PPO agent is defined as a neural network with three fully connected layers. The training loop runs for a specified number of epochs, collecting data from the environment, computing losses, and updating the agent's parameters. The trained model is saved at the end of the training process.

Classes:
    PPOAgent: Defines the neural network architecture for the PPO agent.

Functions:
    forward(state): Performs a forward pass through the network.

Usage:
    Run this script to train the PPO agent. The trained model will be saved as `ppo_agent_final.pth`.

Example:
    $ python train.py

Tensorboard:
   $ tensorboard --logdir=<logdir>
"""

# Add file directory to system path and change working directory (to maintain imports)
import math
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
os.chdir(str(Path(__file__).parent.parent.resolve()))

import os
import time
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from RL.params import *
from RL.env import MyEnv  # Import your custom environment module

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Define hyperparameters
n_steps = 4096
batch_size = 64
n_epochs = 10
gamma = 0.99
### DAVID's EDIT START ###
learning_rate = 3e-3
### DAVID's EDIT END ###
clip_range = 0.18
gae_lambda = 0.95
ent_coef = 0.06
vf_coef = 0.5
max_grad_norm = 0.5
model_name = "simple_navigation"
folder_path = os.path.join(r"RL\training", model_name)
model_path = os.path.join(folder_path, "final")
log_dir = os.path.join(folder_path, "logs")
training_timesteps = 300_000

train_from_scratch = True
load_script_path = "RL/training/simple_navigation/best_model.zip"     

def make_env():
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
        record=False,
        video_name="Current",
    )
    return env
        
# Evaluation environment (not vectorized)
eval_env = Monitor(make_env())

# Create the evaluation callback
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=folder_path,
    log_path=log_dir, 
    eval_freq=5000,  # Evaluate every 5000 timesteps
    deterministic=True,
    render=False
)

# Wrap the environment in a vectorized environment
num_envs = 4

# Create the vectorized environment
vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
# vec_env = make_vec_env(lambda: env, n_envs=1)

# Create the PPO agent with MultiInputPolicy
if train_from_scratch:
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        learning_rate=learning_rate,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir
    )
else:
    model = PPO.load(load_script_path, make_env())
    model.set_env(make_env())

print(f"Tensorboard logs:\ntensorboard --logdir={os.path.join(os.getcwd(), log_dir)}\n")

max_episode_length = make_env().end_time/make_env().time_step
print(f"Maximum Episode Length: {max_episode_length} time-steps\n")
print(f"Minimum Number of Training Episodes: {round(training_timesteps/max_episode_length)}\n")
        
# Start the timer
start_time = time.time()

# Train the PPO agent
model.learn(total_timesteps=training_timesteps, callback=eval_callback)

# Save the trained model
model.save(model_path) # pickle protocol 4 for running model on python 3.6 (modified stable baselines)
print("Model saved at the end of training")

# End the timer and print the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken for training: {total_time:.2f} seconds")

print(f"Tensorboard logs:\ntensorboard --logdir={os.path.join(os.getcwd(), log_dir)}\n")
