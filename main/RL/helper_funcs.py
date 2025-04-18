"""
This module contains various helper functions used throughout the maritime simulation.
"""

import math as mt
import numpy as np

def knotstoms(knots):
    return knots * 0.514444


def mstoknots(ms):
    return ms / 0.514444


# works for coordinates not near to the poles
def longlat_to_xy(longlat):
    "Converts (long, lat) to (x,y) (metres)"
    R = 6371e3
    x = R * mt.radians(longlat[0])
    y = R * mt.log(mt.tan(mt.pi / 4 + mt.radians(longlat[1]) / 2))
    return x, y


def xy_to_longlat(xy):
    "Converts (x,y) (metres) to (long, lat)"
    R = 6371e3
    long = mt.degrees(xy[0] / R)
    lat = mt.degrees(2 * mt.atan(mt.exp(xy[1] / R)) - mt.pi / 2)
    return long, lat


def long_to_x(long):
    "Converts longitude to x"
    R = 6371e3
    x = R * mt.radians(long)
    return x


def lat_to_y(lat):
    "Converts latitude to y"
    R = 6371e3
    y = R * mt.log(mt.tan(mt.pi / 4 + mt.radians(lat) / 2))
    return y


def x_to_long(x):
    "Converts x to longitude"
    R = 6371e3
    long = mt.degrees(x / R)
    return long


def y_to_lat(y):
    "Converts y to latitude"
    R = 6371e3
    lat = mt.degrees(2 * mt.atan(mt.exp(y / R)) - mt.pi / 2)
    return lat


def longlat_midpoint(longlat1, longlat2):
    "returns (long, lat) of the midpoint between two (long, lat) points"
    x1 = longlat_to_xy(longlat1)[0]
    y1 = longlat_to_xy(longlat1)[1]
    x2 = longlat_to_xy(longlat2)[0]
    y2 = longlat_to_xy(longlat2)[1]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return xy_to_longlat((x, y))


def compass_to_math_angle(compass_heading):
    "Converts compass heading to mathematical angle convention (angle from positive x axis)"
    return (90 - compass_heading) % 360


def math_angle_to_compass(math_angle):
    "Converts mathematical angle to compass heading convention (angle from north)"
    return (90 - math_angle) % 360


def heading_to_goal(agent_longlat, goal_longlat):
    "returns compass heading in degrees from agent to goal"
    x1 = longlat_to_xy(agent_longlat)[0]
    y1 = longlat_to_xy(agent_longlat)[1]
    x2 = longlat_to_xy(goal_longlat)[0]
    y2 = longlat_to_xy(goal_longlat)[1]
    math_angle = mt.degrees(mt.atan2(y2 - y1, x2 - x1))
    # Convert to navigational heading
    compass_heading = math_angle_to_compass(math_angle)
    return compass_heading

def random_sample(interval1, interval2):
    "Returns a random sample from one of the two intervals, each of the form (start_number, end_number)"
    intervals = [interval1, interval2]
    chosen_interval = intervals[np.random.choice(len(intervals))]
    return np.random.uniform(chosen_interval[0], chosen_interval[1])
