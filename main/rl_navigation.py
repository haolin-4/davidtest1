"""
rl_navigation.py

Deploys the RL model and creates the an instance of the RL environemnt MyEnv class for visualising the environment.

*(For future works: 
    Create a script that deploys the model without the environment.
    This script is just for visualising the environment
)

"""

import os
import dill

from config import *
from utils.general import get_logger, get_socket, Bool_Step_Checker, RL_Data

from RL.helper_funcs import * 

from stable_baselines3 import PPO
from RL.env import MyEnv  # Import your custom environment module
from RL.params import *

logger = get_logger(os.path.splitext(os.path.basename(__file__))[0], "DEBUG")

# ZMQ sockets
fc_sub = get_socket(PORTS["FC_PUB"], "SUB")                     # Subscribe to flight controller data
obj_procss_sub = get_socket(PORTS["OBJ_PROCESS_PUB"], "SUB")    # Subscribe to processed object data
rl_pub = get_socket(PORTS["RL_PUB"], "PUB")                     # Publish RL actions

def make_env():
    """Create instance of RL environment. Called when jetboard enters autonomous mode."""
    
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
        no_of_generated_obs=no_of_generated_obs,
        random_goal_position_status=random_goal_position_status,
        
        simulation_status=False,
        record=False,
        video_name="Current",
    )
    return env

if __name__ == "__main__":

    # try:
        logger.info("Started main program")         
        
        # Load the trained model
        model = PPO.load(RL_MODEL)

        autonomous = Bool_Step_Checker(initial_bool_value=False) # autonomous mode flag for RL based navigation
        
        logger.info("FYI: Set channel 8 to HIGH for Autonomous Navigation")
        
        # Visualize the model
        while True:
            
            # Load flight controller data from FC interface script
            fc_data = dill.loads(fc_sub.recv())
            sensor_data = fc_data.sensor_data
            mission_waypoints_list = fc_data.mission_waypoints
            autonomous.set(sensor_data.autonomous_status) 
            
            # Load obstacle data from YOLO object detection script
            processed_obs_list = dill.loads(obj_procss_sub.recv())
            
            
            if autonomous.value == True:
                
                if autonomous.PGT == True:
                    logger.info(f"Began autonomous mode. Starting RL environment instance.")
                    waypt_index = 0 
                    env = make_env()
                    observation, _ = env.reset() # reset the environment
                    env.update_ops_env([sensor_data.long, sensor_data.lat], mission_waypoints_list[waypt_index][:2])
                    done_once = False
                    logger.info(f"RL environment instance initialisation completed.")
                
                
                print(processed_obs_list)
                # Get the model's  actions
                action, _states = model.predict(observation, deterministic=True)
                # Update the RL environment 
                observation, dist_to_goal = env.external_update(sensor_data, processed_obs_list, mission_waypoints_list[waypt_index][:2], action)
                
                # Update environment when  waypoint has been reached
                if dist_to_goal < proximity_to_goal:
                    waypt_index += 1
                    
                    if waypt_index >= len(mission_waypoints_list): waypt_index = 0
                    env.update_ops_env([sensor_data.long, sensor_data.lat],
                                    mission_waypoints_list[waypt_index][:2])
                    
                normalized_acc, normalized_yaw_rate = action     

                # Send the commands to the flight controller
                rl_data = RL_Data(normalized_acc=normalized_acc, normalized_yaw_rate=normalized_yaw_rate)
                rl_pub.send(dill.dumps(rl_data))
                    
                # Visualise the environment
                env.render()

            else:
                
                if autonomous.NGT:
                    logger.info(f"Stopped autonomous mode. Closing RL environment instance.")
                    env.close()
                    logger.info(f"Waiting for Autonomous Mode. Set Channel 8 to HIGH for Automous Navigation")

            # logger.info(f"RL Agent Action: {action}")
            
    # except Exception as E:
    #     logger.error(f"Exception occured: {E}")
    #     logger.info("Stopping program...")

    # finally:
    #     env.close()

    #     logger.info("Ended main program")
