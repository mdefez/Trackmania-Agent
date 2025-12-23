from setup.trackmania.env import TMEnv

keys = ["W", "A", "S", "D"]

import random
import gymnasium as gym

import pandas as pd
from .ref_line import line_ref_loss_world, _curve_feature_world, end_racing_line_loss_world, _world_to_block, distance_and_angle_to_racing_line_blocks

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

MAX_DIST_TO_LINE = 0.31
MAX_ELAPSED_TIME = 30.0 * 1000

class RefLineEnv(TMEnv) :
    def __init__(self, track_path_csv: str) :
        super().__init__(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32))
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
        self.start_y = obs[1]
        self.start_time = None
        time.sleep(1.5)

        self.elapsed_time = 0
        self.prev_distance_to_end = None
        return obs, _info
    
    def _get_info(self):
        return {}

    def _is_terminated(self, data):
        if self.distance_to_line > MAX_DIST_TO_LINE :
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
        dist_next_curve, curve_angle = _curve_feature_world(position, self.racing_line)
        # print("Curve angle:", curve_angle)
        dist_next_curve = 0 if abs(dist_next_curve) < 1e-6 else dist_next_curve
        curve_angle = 0 if abs(curve_angle) < 1e-6 else curve_angle
    
        self.time = vdata["time"]
        if not self.start_time :
            self.start_time = self.time
        else :
            self.elapsed_time = self.time - self.start_time

        self.speed = vdata["speed"]
        
        self.x = position[0]

        car_heading_vec = np.array(vdata["direction"], dtype=np.float32).reshape(-1)  # shape (3,)
        car_heading_angle = np.arccos(np.dot(car_heading_vec, np.array([0,0,1])) / (np.linalg.norm(car_heading_vec) * np.linalg.norm(np.array([0,0,1])))) +  np.pi/2

        ## Median line
        d, _, _, _, theta = distance_and_angle_to_racing_line_blocks(_world_to_block(position), car_heading_angle, self.racing_line)
        distance_to_median_line = d
        angle_to_median_line = theta + np.pi/2

        position = np.array(position, dtype=np.float32).reshape(-1)      # shape (2,)
        self.y = position[1]
        self.x = position[0]
        # print("Car position:", position)
        direction = np.array(vdata["direction"], dtype=np.float32).reshape(-1)  # shape (3,)
        # print("Car direction vector:", direction)
        dist_next_curve = np.array([dist_next_curve], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Distance to next curve:", dist_next_curve)
        angle_next_curve = np.array([curve_angle], dtype=np.float32).reshape(-1)    # shape (1,)
        # print("Angle to next curve:", angle_next_curve)
        distance_to_median_line = np.array([distance_to_median_line], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Distance to median line:", distance_to_median_line)
        angle_to_median_line = np.array([angle_to_median_line], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Angle to median line:", angle_to_median_line)
        car_heading_angle = np.array([car_heading_angle], dtype=np.float32).reshape(-1)  # shape (1,)
        # print("Car heading angle (radians):", car_heading_angle)
        
        debug_data = True
        if debug_data :
            print("Position:", position)
            print("Direction:", direction)
            print("Dist to next curve:", dist_next_curve)
            print("Angle to next curve:", angle_next_curve)
            print("Distance to median line:", distance_to_median_line)
            print("Angle to median line:", angle_to_median_line)
            print("Car heading angle:", car_heading_angle)
        

        obs_vector = np.concatenate([
            position,
            direction,
            dist_next_curve,
            angle_next_curve,
            distance_to_median_line,
            angle_to_median_line,
            car_heading_angle
        ], axis=0)

        print()
        
        

        # print(obs)
        # self.history.append(obs)
        return obs_vector


    def reward(self, data) -> float :
        
        position = data[:2]
        x = position[0]
        y = position[1]

        # print("Car position:", _world_to_block([x, y]))

        car_heading = data[2:5]

        # Compute direction from direction vector
        # print("Car heading vector:", car_heading)
        # print("Car heading vector:", car_heading)

        car_heading = np.arccos(np.dot(car_heading, np.array([0,0,1])) / (np.linalg.norm(car_heading) * np.linalg.norm(np.array([0,0,1])))) +  np.pi/2
        # print("Car angle, radians:", car_heading)

        dist = end_racing_line_loss_world([x, y], self.racing_line)
        if self.prev_distance_to_end != None :
            d_dist = dist - self.prev_distance_to_end
        else :
            d_dist = 0.0

        self.prev_distance_to_end = dist

        d, _, _, _, theta = distance_and_angle_to_racing_line_blocks(_world_to_block(position), car_heading, self.racing_line)

        ## Termination penalty
        self.distance_to_line = d
        if self.distance_to_line > MAX_DIST_TO_LINE :
            return - self.speed * 10

        if self.y < self.start_y:
            return -100
        
        if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
            return -5000  
        
        ## Normaliation and combination
        d_dist_term = - d_dist * 300
        # d_dist_term = min(max(d_dist_term, -1e3), 1e3)

        self.speed = min(max(self.speed, 0), 100)
        speed_term = self.speed / 5

        dist_to_line_term = 10 - float(d * 15)
        angle_to_line_term = 10 - float(abs(theta) * 20) ** 2

        print("Distance to line:", d)
        print("Angle to line:", theta)

        print(f"dist_term: {dist_to_line_term:.3f}, angle_term: {angle_to_line_term:.3f}, d_dist term: {d_dist_term:.3f}, speed_term: {speed_term:.3f}")
        return d_dist_term * 20 + angle_to_line_term * speed_term +  dist_to_line_term / 5 * speed_term
        
