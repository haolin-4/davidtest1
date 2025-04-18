"""
obj_inference.py

What it does:
1. Receives boat sensor data from flight controller
2. Receives detected obstacles from YOLO
3. Applies depth perception algorithm and kalman filter to obtain obstacle position, size, and velocity
4. Publishes data to PORTS["OBJ_PROCESS_PUB"] tcp port

"""

import os

import time 
import numpy as np
from math import *

import dill

from utils.general import get_logger, get_socket, Processed_Obstacle_Data, Sensor_Data
from config import *
from filterpy.kalman import KalmanFilter
from RL.helper_funcs import compass_to_math_angle, longlat_to_xy

file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = get_logger(file_name, "INFO")
zmq_logger = get_logger("zmq_log", "CRITICAL")

# establish sockets
yolo_sub = get_socket(PORTS["YOLO_PUB"], "SUB")      # Receive detected objs
fc_sub = get_socket(PORTS["FC_PUB"], "SUB")          # Receive FC data
obj_process_pub = get_socket(PORTS["OBJ_PROCESS_PUB"], "PUB")      # Publish processed obj data

# Class for tracking location, heading and velocity of detected objects
class Object_tracker():
    
    # Jetboard physical attributes (Used for calculating object depth)
    ver_fov = CAMERA['FOV'][0]
    hor_fov = CAMERA['FOV'][1]
    cam_elevation = CAMERA['elevation']
    cam_resolution = RESIZED_VID_RESOLUTION    
    
    # Kalman Filter Properties:
    
    # Time step
    dt = 1
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    
    # State Transition Matrix (F)
    F = np.array([[1,0,dt,0],
                [0,1,0,dt],
                [0,0,1,0],
                [0,0,0,1]])

    # Measurement function
    H = np.array([[1,0,0,0],
                [0,1,0,0]])
    # Defines how the measurement relates to the state matrix


    # Initial state covariance (P)
    P = np.eye(4) * 500   # Large uncertainty in initial state

    # Measurement noise covariance (R)
    proposed_measurement_noise_std = 50.0     # standard deviation
    # actual_measurement_noise_std = 1.0 
    R = np.eye(2) * proposed_measurement_noise_std**2

    # Process Noise Covariance (Q)
    process_noise_std = 1.0
    Q = np.array([  [dt4/4,     0, dt3/2,     0],
                    [    0, dt4/4,     0, dt3/2],
                    [dt3/2,     0,   dt2,     0],
                    [    0,   dt3,     0,   dt2]]) * process_noise_std**2

    def __init__(self, id, cls, bbox):
        
        # Initialise Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # Define shape of state and measurement vector i.e x=[x, y, x_dot, y_dot], z=[x,y] 

        # Define necessary matrices
        self.kf.F = self.F
        self.kf.H = self.H
        self.kf.P = self.P
        self.kf.R = self.R
        self.kf.Q = self.Q
    
        self.id = id
        self.cls = cls
        self.bbox = bbox
        
        self.xy_rel_raw, self.size = self.calculate_position_and_size()
        
        # Initial state vector
        self.kf.x = np.append(self.xy_rel_raw, [0,0])      # Define initial position and velocity of object
        
        # self.filtered_position_list = []
        # self.raw_position_list = []
        
    def calculate_position_and_size(self):
        """Estimates the object's coordinates and size based on the objects bounding box.
        
            Returns:
                xy_rel: numpy.ndarray
                    Relative coordinates of object in metres. (w.r.t camera)
                size: float
                    Horizontal size of object in metres. Estimated based on bbox width.
        """
        
        x1, y1, x2, y2 = self.bbox
        x, w = (x2+x1)/2, x2-x1 
        
        img_w, img_h = self.cam_resolution
        horizon_lvl = img_h / 2
        
        if y2 - horizon_lvl > 0:
            y_rel = (self.cam_elevation * horizon_lvl) / ((y2-horizon_lvl) * tan(radians(self.ver_fov/2)))
        else:
            y_rel = nan  # Account for situation where bottom of obj bbox is above or on the horizon line
        
        x_rel = (x/img_w-0.5) * (y_rel*tan(radians(90-self.hor_fov/2)))
        xy_rel = np.array([x_rel, y_rel])
        
        size = (w/img_w) * (y_rel*tan(radians(90-self.hor_fov/2))*2)
    
        return xy_rel, size
 
    def update_tracker(self, bbox):
        """Update the Kalman Tracker"""
        
        self.bbox = bbox
        self.xy_rel_raw, self.size = self.calculate_position_and_size() # in metres
        
        self.kf.predict()    # Predict the next state
        self.kf.update(self.xy_rel_raw)  # Update with the new measurement
        
        # self.raw_position_list.append(self.xy_rel_raw)
        # self.filtered_position_list.append(self.kf.x[:2])
    
    @staticmethod  
    def get_transformation_matrix(boat_math_angle_rad, vector):
        """Returns transformation matrix for converting to absolute values"""
        return np.array([[ np.sin(boat_math_angle_rad),  np.cos(boat_math_angle_rad), vector[0]],
                        [-np.cos(boat_math_angle_rad),  np.sin(boat_math_angle_rad), vector[1]],
                        [                           0,                            0,           1]])
        
        
    def calculate_absolute_values(self, boat_heading: float, boat_xy: np.ndarray, boat_velocity: np.ndarray):
        """Returns obs absolute xy position and velocity"""

        boat_math_angle_rad = np.deg2rad(compass_to_math_angle(boat_heading))
        
        # Create obj pos vector for matrix calculations
        obj_xy_rel_to_cam = np.array([
                                        self.kf.x[0], 
                                        self.kf.x[1],
                                        1
                                        ]) 
        
        # Create velocity vector for matrix calculations
        obs_vel_rel_to_cam = np.array([ self.kf.x[2],
                                        self.kf.x[3],
                                        1])
        
        # Get camera's absolute xy coordinates 
        cam_xy = boat_xy + JETBOARD['FC_to_cam_dist'] * np.array([np.cos(boat_math_angle_rad),
                                                                    np.sin(boat_math_angle_rad)])
        # Calulate absolute position of obs
        self.xy_abs = np.matmul(self.get_transformation_matrix(boat_math_angle_rad, cam_xy), 
                           obj_xy_rel_to_cam)[:2] 
        
        # Calculate absolute velocity of obs
        self.vel_abs = np.matmul(self.get_transformation_matrix(boat_math_angle_rad, boat_velocity), 
                                obs_vel_rel_to_cam)[:2]
        

    def get_obs_data(self):
        return Processed_Obstacle_Data(
                            id=self.id,
                            cls=self.cls,
                            bbox=self.bbox,
                            xy_rel_raw=self.xy_rel_raw,
                            xy_rel_filtered=self.kf.x[:2],
                            xy_abs=self.xy_abs,
                            velocity_rel=self.kf.x[2:],
                            velocity_abs=self.vel_abs,
                            size=self.size
                            )

if __name__ == "__main__":

    # try:
        logger.info("Started main program") 

        prev_t = 0
        tracked_obs_dict = {} 

        while True:

            # Load flight controller data 
            fc_data = dill.loads(fc_sub.recv())
            sensor_data = fc_data.sensor_data
            
            # sensor_data = Sensor_Data(
            #     longitude=0,
            #     latitude=0,
            #     altitude=0,
            #     pitch=0,
            #     roll=0,
            #     yaw=0,
            #     heading=90,
            #     velocity=np.array([1,0]),
            #     battery_current=0,
            #     battery_voltage=0,
            #     mode="MANUAL",
            #     armed=True,
            #     gps_fix=True,
            #     channels_list=[1500 for i in range(1, 8+1)],
            #     autonomous_status=True
            #                 )

            # Load detected obstacle list
            detections_list = dill.loads(yolo_sub.recv())

            time_elapsed_secs = time.time() - prev_t
            Object_tracker.dt = time_elapsed_secs               # Update the KF time step 

            # Infer object position and velocity and update the tracker
            processed_obs_list = []
            closest_distance = np.inf
            closest_dist_obs_id = -1
            for obj in detections_list:
                                                                                                                                                                                                                                                                                                                                                                                    
                obj_id = obj.id
                
                if obj_id not in tracked_obs_dict:  # Track objects if not done already
                    tracked_obs_dict[obj_id] = Object_tracker(obj_id, obj.cls, obj.bbox)
                
                tracked_obs_dict[obj_id].update_tracker(bbox=obj.bbox)  # Update kalman filter
                tracked_obs_dict[obj_id].calculate_absolute_values(
                                                                    boat_heading=sensor_data.heading,
                                                                    boat_xy=np.array(longlat_to_xy((sensor_data.long, sensor_data.lat))),
                                                                    boat_velocity=np.array(sensor_data.velocity)
                                                                    )
                
                # Find nearest obstacle 
                obs_distance_to_cam = np.linalg.norm(tracked_obs_dict[obj_id].kf.x[:2])
                if obs_distance_to_cam <= closest_distance:
                    closest_distance = obs_distance_to_cam
                    closest_dist_obs_id = obj_id

                processed_obs_list.append(tracked_obs_dict[obj_id].get_obs_data())            
            
            obj_process_pub.send(dill.dumps(processed_obs_list))   # Publish processed objects
            
            logger.info(f"Number of obstacles: {len(processed_obs_list)}, Closest obstacle: Obs {closest_dist_obs_id} {closest_distance:.2f}m to camera") 

    # except Exception as E:
    #     logger.error(f"Exception occured: {E}")
    #     logger.info("Stopping program...")

    # finally:
    #     logger.info("Ended main program")

