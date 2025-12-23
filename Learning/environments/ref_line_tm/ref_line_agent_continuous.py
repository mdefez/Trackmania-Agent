from setup.trackmania.env import TMEnv

keys = ["W", "A", "S", "D"]

import random
import gymnasium as gym

import pandas as pd
from .continuous_track import TrackmaniaTrack


import requests
url = "http://127.0.0.1:8080/api/data"
headers = {"Content-Type": "application/json"}

def send_data(x, y, direction):
    payload = {
        "vehicleData": {
            "position": [float(x),float(y)],
            "direction": float(direction)
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response


import math
import time
import numpy as np

MAX_DIST_TO_LINE = 10
MAX_ELAPSED_TIME = 20.0 * 1000

data = """Block,X,Y,Z,Rotate,cx,cy,cz
RoadTechStart,16,1,21,None,528.0,12.0,688.0
RoadTechStraight,16,1,22,None,528.0,12.0,720.0
RoadTechStraight,16,1,23,None,528.0,12.0,752.0
RoadTechStraight,16,1,24,None,528.0,12.0,784.0
RoadTechCurve1,16,1,25,Right,528.0,12.0,816.0
RoadTechStraight,15,1,25,None,496.0,12.0,816.0
RoadTechStraight,14,1,25,None,464.0,12.0,816.0
RoadTechCurve1,13,1,25,Left,432.0,12.0,816.0
RoadTechStraight,13,1,26,None,432.0,12.0,848.0
RoadTechStraight,13,1,27,None,432.0,12.0,880.0
RoadTechCurve1,13,1,28,Left,432.0,12.0,912.0
RoadTechStraight,14,1,28,None,464.0,12.0,912.0
RoadTechStraight,15,1,28,None,496.0,12.0,912.0
RoadTechFinish,16,1,28,None,528.0,12.0,912.0"""

class RefLineEnvContinuous(TMEnv) :
    def __init__(self, track_path_csv: str) :
        super().__init__(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32))
        self.history = []
        self.track_path = track_path_csv

        df = pd.read_csv(self.track_path, sep=',')
        self.racing_line = df[['X','Z']].to_numpy(dtype=np.float32)
        # Add 16 offset to center of block to all X
        # print(self.racing_line)
        self.racing_line[:,0] += 0.5
        self.racing_line[:,1] += 0.5

        self.prev_distance_to_end = np.inf

        self.start_time = None

        self.track = TrackmaniaTrack(data)
        self.track.compute_pivots()
    
    def _action_to_key(self, action):
        # print(action)
        keys = []
        if action[0] > 0.1 :
            keys.append("W")
        if action[1] < -0.1 :
            keys.append("A")
        if action[1] > 0.1 :
            keys.append("D")
        if action[2] > 0.1 :
            keys.append("S")
        return keys
    
    def _compute_reward(self, data):
        reward = self.reward(data)
        print("reward :", reward)
        return reward
    
    def reset(self, seed=None, options=None):
        obs, _info = super().reset(seed, options)
        self.start_y = 688
        self.start_time = None
        time.sleep(1.5)

        self.elapsed_time = 0
        self.prev_distance_to_end = None
        return obs, _info
    
    def _get_info(self):
        return {}

    def _is_terminated(self, data):
        if self.distance_to_median_line > MAX_DIST_TO_LINE :
            print("Terminated: too far from line")
            return True
        if self.y < self.start_y:
            print("Terminated: went backwards")
            return True
        if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
            print("Terminated: time out")
            return True

        return False
    
    def _is_truncated(self, data):
        return False
    
    def _data_to_observation(self, data):
        vdata = data["vehicleData"]
        position = [vdata["position"][0], vdata["position"][2]]

        ## Position features
        self.x = position[0]
        self.y = position[1]
        self.speed = vdata["speed"]

        ## Track
        line_angle = self.track.compute_line_angle(position)
        line_dist = self.track.distance_to_median_line(position)
        line_dist = 0 if abs(line_dist) < 1e-6 else abs(line_dist)
        next_curve_dist = self.track.distance_to_next_curve(position)
        next_curve_angle = self.track._next_curve_angle(position)

        next_curve_dist = 0 if abs(next_curve_dist) < 1e-6 else next_curve_dist
        next_curve_angle = 0 if abs(next_curve_angle) < 1e-6 else next_curve_angle

        ## Car features
        car_heading_vec = np.array(vdata["direction"], dtype=np.float32).reshape(-1)  # shape (3,)
        car_heading_angle = np.arccos(np.dot(car_heading_vec, np.array([-1,0,0])) / (np.linalg.norm(car_heading_vec) * np.linalg.norm(np.array([1,0,0]))))

        # Time
        self.time = vdata["time"]
        if not self.start_time :
            self.start_time = self.time
        else :
            self.elapsed_time = self.time - self.start_time

        
        ## Median line
        angle_to_median_line = line_angle - car_heading_angle

        position = np.array(position, dtype=np.float32).reshape(-1)      # shape (2,)
        # print("Car position:", position)
        direction = np.array(vdata["direction"], dtype=np.float32).reshape(-1)  # shape (3,)
        # print("Car direction vector:", direction)
        dist_next_curve = np.array([next_curve_dist], dtype=np.float32).reshape(-1) # shape (1,)
        # print("Distance to next curve:", dist_next_curve)
        angle_next_curve = np.array([next_curve_angle], dtype=np.float32).reshape(-1)    # shape (1,)
        # print("Angle to next curve:", angle_next_curve)
        distance_to_median_line = np.array([line_dist], dtype=np.float32).reshape(-1) # shape (1,)
        # print("Distance to median line:", distance_to_median_line)
        angle_to_median_line = np.array([angle_to_median_line], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Angle to median line:", angle_to_median_line)
        car_heading_angle = np.array([car_heading_angle], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Car heading angle (radians):", car_heading_angle)
        speed = np.array([self.speed], dtype=np.float32).reshape(-1)  # shape (1,)

        self.car_heading_angle = car_heading_angle[0]
        self.angle_to_median_line = angle_to_median_line[0]
        self.distance_to_median_line = distance_to_median_line[0]
        self.angle_curve_next = angle_next_curve[0]
        self.distance_to_curve_next = dist_next_curve[0]
        self.direction_vector = direction
        self.position_vector = position

        debug_data = False
        if debug_data :
            # print("Position:", position)
            # print("Direction:", direction)
            print("Dist to next curve:", dist_next_curve)
            print("Angle to next curve:", angle_next_curve)
            print("Distance to median line:", distance_to_median_line)
            print("Angle to median line:", angle_to_median_line)
            print("Car heading angle:", car_heading_angle)
            print("Speed:", speed)


        obs_vector = np.concatenate([
            # position, # 2
            # direction, # 3
            dist_next_curve / 200, # 1
            angle_next_curve, # 1
            distance_to_median_line / 10, # 1
            angle_to_median_line, # 1
            car_heading_angle, # 1,
            speed
        ], axis=0)


        print()
        
        

        # print(obs)
        # self.history.append(obs)
        return obs_vector


    def reward(self, data) -> float :
        
        position = self.position_vector
        dist = self.track.compute_distance_to_finish(position)
        # print("Distance to finish:", dist)
        if self.prev_distance_to_end != None :
            d_dist = dist - self.prev_distance_to_end
        else :
            d_dist = 0.0
        self.prev_distance_to_end = dist


        median_line_angle = self.angle_to_median_line
        # print("Median angle", median_line_angle)

        ## Termination penalty
        distance_to_line = abs(self.distance_to_median_line)
        # print("Distance to line:", self.distance_to_line)

        if distance_to_line > MAX_DIST_TO_LINE :
            return - abs(self.angle_to_median_line) * 10

        if self.y < self.start_y:
            print("Went backwards")
            print("y:", self.y, "start_y:", self.start_y)
            return -1000
        
        if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
            print("Time out")
            return -5000  
        
        ## Normaliation and combination
        d_dist_term = - d_dist * 100
        dist_to_finish_term = - self.track.compute_distance_to_finish(position)

        # d_dist_term = min(max(d_dist_term, -1e3), 1e3)

        self.speed = min(max(self.speed, 0), 100)
        speed_term = self.speed / 5

        dist_to_line_term = - float(abs(self.distance_to_median_line) * 3)
        # dist_to_line_term = 0
        angle_to_line_term = - float(abs(median_line_angle) * 3)

        # print("Distance to line:", d)
        # print("Angle to line:", median_line_angle)

        # print(f"dist_term: {dist_to_line_term:.3f}, angle_term: {angle_to_line_term:.3f}, d_dist term: {d_dist_term:.3f}, speed_term: {speed_term:.3f}")
        return angle_to_line_term + dist_to_line_term + d_dist_term
        
