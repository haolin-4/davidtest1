import numpy as np
import time

class Bool_Step_Checker:

    def __init__(self, initial_bool_value):

        self.got_change = False
        self.PGT = False # Positive going transition
        self.NGT = False # Negative going transition

        self.prev_val = initial_bool_value
        self.value = initial_bool_value

    def set(self, new_value):
        """Set the variable to a new value. State change is checked after setting value."""
        
        self.prev_val = self.value
        self.value = new_value
        
        if self.value != self.prev_val:   self.got_change = True
        else:   self.got_change = False
        
        if self.value and not self.prev_val: self.PGT = True
        else: self.PGT = False
        
        if not self.value and self.prev_val: self.NGT = True
        else: self.NGT = False

class ZMQ_Data:
    
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    def __repr__(self):
        return

class Sensor_Data(ZMQ_Data):
    
    def __init__(self, 
                 longitude: float,
                 latitude: float,
                 altitude: float,
                 pitch: float,
                 roll: float,
                 yaw: float,
                 heading: float,
                 velocity: tuple,
                 battery_voltage: float,    
                 battery_current: float,    
                 mode: int,
                 armed: bool,
                 gps_fix: int,
                 channels_list: list,
                 autonomous_status: bool):
    
        self.long = longitude
        self.lat = latitude
        self.alt = altitude
        self.pitch = pitch
        self.roll = roll,
        self.yaw = yaw
        self.heading = heading
        self.velocity = velocity
        self.battery_current = battery_voltage
        self.battery_current = battery_current
        self.armed = armed
        self.mode = mode
        self.gps_fix =  gps_fix
        self.channels_list = channels_list
        self.autonomous_status = autonomous_status
        
        self.time_stamp = time.time()
        
    
class FC_Data(ZMQ_Data):
    
    def __init__(self, sensor_data, mission_waypoints: list):
        
        self.sensor_data = sensor_data
        self.mission_waypoints = mission_waypoints
        
        self.time_stamp = time.time()
        
class RL_Data(ZMQ_Data):
    
    def __init__(self, normalized_acc, normalized_yaw_rate):
        
        self.normalized_acc = normalized_acc
        self.normalized_yaw_rate = normalized_yaw_rate
        
        self.time_stamp = time.time()
    
class Processed_Obstacle_Data():
    
    def __init__(self, 
                 id, 
                 cls, 
                 bbox, 
                 xy_rel_raw, 
                 xy_rel_filtered,
                 xy_abs,
                 velocity_rel,
                 velocity_abs,
                 size):
        
        self.id = id 
        self.cls = cls  
        self.bbox = bbox  
        self.xy_rel_raw = xy_rel_raw  
        self.xy_rel_filtered = xy_rel_filtered 
        self.xy_abs = xy_abs
        self.velocity_rel = velocity_rel  
        self.velocity_abs = velocity_abs       
        self.size = size
        self.time_stamp = time.time()
        
class Detected_Object_Data():
    
     def __init__(self, 
                  id,
                 cls, 
                 bbox):
        
        self.id = id
        self.cls = cls  
        self.bbox = bbox  
        self.time_stamp = time.time()   
        
def get_logger(name:str, level:str, config:dict =None):
    """Get a logger"""

    import logging

    log_lvl = {"DEBUG": logging.DEBUG,
               "INFO": logging.INFO,
               "WARNING": logging.WARNING,
               "ERROR": logging.ERROR,
               "CRITICAL": logging.CRITICAL,}

    if config is None:
        config = {"format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                  "datefmt": '%d/%m %H:%M:%S'}
    # logger config
    logging.basicConfig(
        format=config["format"],
        datefmt=config["datefmt"],
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_lvl[level])

    return logger 
        
def get_socket(addr:str, type:str):
    """Get a ZMQ socket"""
    
    import zmq
    context = zmq.Context()

    socket_type = {"PUB":zmq.PUB, "SUB":zmq.SUB, "PUSH":zmq.PUSH, "PULL":zmq.PULL, "REQ":zmq.REQ, "REP":zmq.REP}

    socket = context.socket(socket_type[type])
    if type == "SUB": 
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
    elif type == "PULL" or type == "REQ":
        socket.connect(addr)
    else:
        socket.bind(addr)
    
    return socket