"""
This script performs interfacing between the jetson nano and the flight controller. 

Sensor readings and flight path commands are read from the FC and publicised throught the fc_pub socket.
RL agent commands are read in from the pi_pub socket and written to the FC.

To configure permanent sudo access to UART port
1. Add user to dialout group:
    `sudo usermod -aG dialout <username>`
2. Disable `nvgetty` service:
    `sudo systemctl stop nvgetty`
    `sudo systemctl disable nvgetty`
3. Reboot:
    `sudo reboot`
4. Ensure `nvgetty` service is disabled:
    `systemctl status nvgetty`
5. Ensure `dialout` is listed:
    `groups`
    or 
    `ls -l /dev/ttyTHS1`
    
Run with 

"""

# uncomment to give temporary access to the UART port
# import subprocess
# command = ['sudo', 'chmod', '666', '/dev/ttyTHS1']
# subprocess.run(command)

import os

from dronekit import connect, VehicleMode
from pymavlink import mavutil

import numpy as np
import time
import threading

import dill

from config import *
from utils.general import Bool_Step_Checker, Sensor_Data, FC_Data, get_logger, get_socket

if USE_SITL_SIM: import keyboard

file_name = os.path.splitext(os.path.basename(__file__))[0]

# Loggers
logger = get_logger(file_name, "INFO")
read_thread_logger = get_logger(file_name+': Read-Thread', "INFO")
write_thread_logger = get_logger(file_name+': Write-Thread', "CRITICAL")

# ZMQ sockets
fc_pub = get_socket(PORTS['FC_PUB'], "PUB")
pc_to_jetson_pub = get_socket(PORTS['PC_TO_JETSON_PUB'], "PUB")
rl_sub = get_socket(PORTS['RL_PUB'], "SUB")

# variables
#---------------------------------------------------------------------------------
autonomous = Bool_Step_Checker(initial_bool_value=False)  # Autonomous mode flag for RL based navigation

# Create object to connect to the flight controller
fc_port = SITL_PORT if USE_SITL_SIM else JETSON_UART_PORT
vehicle = connect(ip=fc_port, baud=FC_BAUD_RATE, wait_ready=True, timeout=FC_TIMEOUT)
#---------------------------------------------------------------------------------

# Define a callback to listen to servo output messages from simulation
sim_servo_pwm =vehicle.parameters[f'SERVO{SERVO_CHANNEL}_TRIM']
sim_throttle_pwm = vehicle.parameters[f'SERVO{THROTTLE_CHANNEL}_TRIM']
@vehicle.on_message('SERVO_OUTPUT_RAW')
def listener(self, name, message):
    global sim_servo_pwm, sim_throttle_pwm 

    sim_servo_pwm = message.servo1_raw
    sim_throttle_pwm = message.servo3_raw
       
def set_servo(vehicle, channel, pwm_value):
    """
    Set a servo channel to a specific PWM value.

    Parameters
    ----------
    vehicle: dronekit Vehicle object
        Vehicle used
    channel: int
        Servo channel number on the flight controller (e.g., 5 for SERVO5)
    pwm_value: float
        PWM value (usually between 1000 to 2000) to set the servo position
    """
    
    # Ensure SERVOX_FUNCTION is set to 1 or 0 to allow control of servo through MAVLINK
    vehicle.parameters[f"SERVO{SERVO_CHANNEL}_FUNCTION"] = 1 
    
    msg = vehicle.message_factory.command_long_encode(
        0, 0,                                   # Target system and target component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,   # MAVLink command to set servo
        0,                                      # Confirmation
        channel,                                # Servo channel number
        pwm_value,                              # PWM value (1000 to 2000 microseconds)
        0, 0, 0, 0, 0)                          # Unused parameters

    vehicle.send_mavlink(msg)
    vehicle.flush()
    
def enter_mode(mode):
    """Set the vehicle operation mode"""
    
    vehicle.mode = VehicleMode(mode)
    while vehicle.mode.name != mode:
        read_thread_logger.debug(f"Waiting for {mode} mode...")
        time.sleep(0.5)
    read_thread_logger.info(f"Entered Vehicle {mode} mode")

def get_commands():
    """Read programmed waypoints and vehicle commands."""

    read_thread_logger.info("Waiting for vehicle commmands.")

    # Get programmed waypoints
    vehicle.commands.download()
    vehicle.commands.wait_ready()  # Wait until all commands are downloaded
    
    commands = []
    mission_waypoints_string = ""
    for i, cmd in enumerate(vehicle.commands):
        
        waypoint = [cmd.y,cmd.x,cmd.z]
        if waypoint == [0.0, 0.0, 0.0]: waypoint = [vehicle.home_location.lon, vehicle.home_location.lat, vehicle.home_location.alt]
        
        commands.append(waypoint)
        mission_waypoints_string += f"    Waypoint {i}| Long Lat Alt: {commands[-1]} \n"
    
    read_thread_logger.info("Vehicle commands received. \n    Mission Waypoints:\n"+mission_waypoints_string)

    return commands

def emulate_channel_8():
    """Emulates rc channel 8 signal for simulation.
        Press 'a' once to turn on. Press again to turn off."""
    
    import tkinter as tk
    
    window = tk.Tk()
    window.title("Channel 8 Emulation")
    window.geometry("200x100")  # Small window size
    
    switch_on = False
    def toggle_channel_8():
        nonlocal switch_on
        switch_on = not switch_on
        
        channel_8 = vehicle.parameters['RC8_MAX'] if switch_on else vehicle.parameters['RC8_MIN']
        vehicle.channels.overrides = {'8': int(channel_8)}
    
    # Create a button that calls trigger_fc_action when clicked
    button = tk.Button(window, text="Channel 8", command=toggle_channel_8)
    button.pack(pady=20)  # Add padding for aesthetics

    while True:
        # Start the Tkinter event loop
        window.mainloop()

def setup():
    """Code to be run at the start of the program."""

    enter_mode(mode="AUTO") # Enter guided mode on start

def read_from_FC(): 
    """Cotinuously requests parameter values from the flight controller."""

    global autonomous

    try:
        commands = get_commands() # get vehicle waypoints

        while True:

            # Read sensors
            sensor_data = Sensor_Data(
                longitude=vehicle.location.global_frame.lon,
                latitude=vehicle.location.global_frame.lat,
                altitude=vehicle.location.global_frame.alt,
                pitch=vehicle.attitude.pitch,
                roll=vehicle.attitude.roll,
                yaw=vehicle.attitude.yaw,
                heading=vehicle.heading,
                velocity=vehicle.velocity,
                battery_voltage=vehicle.battery.voltage,
                battery_current=vehicle.battery.current,
                mode=vehicle.mode,
                armed=vehicle.armed,
                gps_fix=vehicle.gps_0.fix_type,
                channels_list=[vehicle.channels[f'{i}'] for i in range(1, 8+1)],
                autonomous_status=autonomous.value
            )

            fc_data = FC_Data(sensor_data=sensor_data,
                                mission_waypoints=commands)
            
            # Set autonomous mode when channel 8 is high
            if vehicle.parameters['RC8_MAX'] == (vehicle.channels.overrides.get('8') if USE_SITL_SIM else vehicle.channels['8']):
                autonomous.set(True)
                if autonomous.PGT: read_thread_logger.info("Began Autonomous Mode.")
            else: 
                autonomous.set(False)
                if autonomous.NGT: read_thread_logger.info("Stopped Autonomous Mode.")
                
            fc_pub.send(dill.dumps(fc_data))
            read_thread_logger.debug("Sending FC data to fc_pub")

    except Exception as E:
        read_thread_logger.error(f"Error in 'read thread': {E}.")

def agent_actions_to_pwm(normalized_acc, normalized_yaw_rate):
    """Convert RL output actions to pwm values for the throttle and servo."""
    
    throttle_min_pwm = vehicle.parameters[f'SERVO{THROTTLE_CHANNEL}_MIN']
    throttle_max_pwm = vehicle.parameters[f'SERVO{THROTTLE_CHANNEL}_MAX']
    
    servo_min_pwm = vehicle.parameters[f'SERVO{SERVO_CHANNEL}_MIN']
    servo_max_pwm = vehicle.parameters[f'SERVO{SERVO_CHANNEL}_MAX']
    servo_neutral_pwm = vehicle.parameters[f'SERVO{SERVO_CHANNEL}_TRIM']
    
    # Assuming linear relationship 
    throttle = normalized_acc * (throttle_max_pwm-throttle_min_pwm) + throttle_min_pwm
    servo = normalized_yaw_rate * (servo_max_pwm-servo_neutral_pwm) + servo_neutral_pwm
    
    return int(throttle), int(servo)

def write_to_FC():
    """Cotinuously write parameter values to the flight controller."""

    try:
        while True:
            
            # Run RL based autonomous navigation
            if autonomous.value:
                if autonomous.PGT:  # Enter manual mode for control of servo and throttle
                    enter_mode('MANUAL')
                    write_thread_logger.info("Writing to sim vehicle in auto mode")
                    
                # Load RL agent's actions
                rl_data = dill.loads(rl_sub.recv())
                
                throttle_pwm, servo_pwm = agent_actions_to_pwm(rl_data.normalized_acc, rl_data.normalized_yaw_rate)
                
                if not USE_SITL_SIM: # Write to flight controller directly
                    set_servo(vehicle, SERVO_CHANNEL, servo_pwm)
                    set_servo(vehicle, THROTTLE_CHANNEL, throttle_pwm)
                    
            # Run regular mission planner based waypoint navigation
            else:
                if autonomous.NGT: # Return jetboard to guided mode
                    print("entering guided mode")
                    enter_mode('GUIDED')
                    write_thread_logger.info("Writing to sim vehicle in guided mode")
                    
             # Publish actions to seperate script on jetson nano if using simulation
            if USE_SITL_SIM:
                write_thread_logger.debug(f"Sending to PORTS['PC_TO_JETSON_PUB']: {[sim_throttle_pwm, sim_servo_pwm]}")
                pc_to_jetson_pub.send(dill.dumps([sim_throttle_pwm, sim_servo_pwm]))                
                
    except Exception as E:
        write_thread_logger.error(f"Error in 'write thread': {E}.")


if __name__ == "__main__":

    try:
        read_thread_logger.info("Started main program") 

        # Run on start up
        setup()

        # Reading and writing is performed simultaneously in different threads
        read_thread = threading.Thread(target=read_from_FC, daemon=True)
        write_thread = threading.Thread(target=write_to_FC, daemon=True)

        read_thread.start()
        read_thread_logger.info("Read thread started")
        write_thread.start()
        write_thread_logger.info("Write thread started")
        
        if USE_SITL_SIM: 
            emulate_channel_8()
    
    except Exception as E:
        logger.error(f"Exception occured: {E}")
        logger.info("Stopping program...")

    finally:
        read_thread.join()
        read_thread_logger.info("Read thread ended")
        write_thread.join()
        write_thread_logger.info("Write thread ended")

        vehicle.close()
        logger.info("Ended main program")
