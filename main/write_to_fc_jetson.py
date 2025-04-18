"""
This script emulates the actions performed by the Mission Planner SITL vehicle on the jetboard.

Actions for the jetboard is communicated from read_write_FC.py on the PC through PC_TO_JETSON port and written to the 
flight controller on the jetboard.

Usage:
- Use this script for demostration with the Mission Planner SITL simulator. 

"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil

import time

import dill
from config import *
from utils.general import get_logger, get_socket

logger = get_logger(__name__, "INFO")

# ZMQ sockets
pc_to_jetson_sub = get_socket(PORTS['PC_TO_JETSON_PUB'], "SUB")

# variables
#---------------------------------------------------------------------------------
# Create object to connect to the flight controller
real_vehicle = connect(ip=JETSON_UART_PORT, baud=FC_BAUD_RATE, wait_ready=True, timeout=FC_TIMEOUT)
#---------------------------------------------------------------------------------

def set_servo(channel, pwm_value):
    """
    Set a servo channel to a specific PWM value.

    Parameters
    ----------
    channel: int
        Servo channel number on the flight controller (e.g., 5 for SERVO5)
    pwm_value: float
        PWM value (usually between 1000 to 2000) to set the servo position
    """
    
    msg = real_vehicle.message_factory.command_long_encode(
        0, 0,                                   # Target system and target component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,   # MAVLink command to set servo
        0,                                      # Confirmation
        channel,                                # Servo channel number
        pwm_value,                              # PWM value (1000 to 2000 microseconds)
        0, 0, 0, 0, 0)                          # Unused parameters

    real_vehicle.send_mavlink(msg)
    real_vehicle.flush()
    

if __name__ == "__main__":

    try:
        logger.info("Started main program") 

        # Run on start up
        mode = "MANUAL"
        real_vehicle.mode = VehicleMode(mode)
        while real_vehicle.mode.name != mode:
            logger.info(f"Waiting for {mode} mode...")
            time.sleep(0.5)
        logger.info(f"Entered Vehicle {mode} mode")

        
        real_vehicle.parameters[f"SERVO{SERVO_CHANNEL}_FUNCTION"] = 1 # Ensure SERVOX_FUNCTION is set to 1 or 0 to allow control of servo through MAVLINK
        set_servo(SERVO_CHANNEL, real_vehicle.parameters[f"SERVO{SERVO_CHANNEL}_TRIM"]) # Set initial servo position to servo trim value
        
        while True:
            
            actions = dill.loads(pc_to_jetson_sub.recv())
            throttle_pwm, servo_pwm = actions
            print(f"Received message: Throttle: {throttle_pwm}, Nozle: {servo_pwm}")
            set_servo(THROTTLE_CHANNEL, throttle_pwm)
            set_servo(SERVO_CHANNEL, servo_pwm)
        
    
    except Exception as E:
        logger.error(f"Exception occured: {E}")
        logger.info("Stopping program...")

    finally:
        real_vehicle.close()
        logger.info("Ended main program")
