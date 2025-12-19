from setup.trackmania.env import TMEnv

keys = ["W", "A", "S", "D"]

import random
import gymnasium as gym

import pandas as pd
from .ref_line import line_ref_loss, distance_to_end_on_racing_line, distance_to_next_curve

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

class RefLineEnv(TMEnv) :
    def __init__(self, track_path_csv: str) :
        super().__init__(observation_space=gym.spaces.Dict({
            "position": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,)),
            "direction": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
            "dist_next_curve": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
            "angle_next_curve": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        }))
        self.history = []
        self.track_path = track_path_csv

        df = pd.read_csv(self.track_path, sep=';')
        self.racing_line = df[['X','Z']].to_numpy()

        self.x = 16
        self.y = 21
        self.direction = math.pi / 2
        send_data(self.x, self.y, self.direction)

        self.prev_distance_to_end = distance_to_end_on_racing_line([self.x, self.y], self.racing_line)


    def reset(self, seed=None, options=None):
        self.x = 16
        self.y = 21
        self.direction = math.pi / 2
        self.controller.reset()
        time.sleep(0.1)
        send_data(self.x, self.y, self.direction)
        return self._get_obs(), self._get_info()
    
    def _action_to_key(self, action):
        self.direction += action[0]
        self.direction = self.direction % (2 * math.pi)
        self.x += math.cos(self.direction) / 5
        self.y += math.sin(self.direction) / 5

        print (self.x, self.y, self.direction)
        print()
        # print(action)
        time.sleep(0.01)
        send_data(self.x, self.y, self.direction)
        return []
    
    def _compute_reward(self, data):
        reward = - self.loss(data)
        # print(reward)
        return reward
    
    def _get_info(self):
        return {"x": self.x, "y": self.y, "direction": self.direction}

    def _is_terminated(self, data):
        if self.distance_to_line > 1 :
            return True
        return False
    
    def _is_truncated(self, data):
        return False
    
    def _data_to_observation(self, data):
        vdata = data["vehicleData"]
        dist_next_curve, curve_angle = distance_to_next_curve(vdata["position"], self.racing_line)
        obs = {
            "position": np.array(vdata["position"], dtype=float),
            "direction": np.array([vdata["direction"]], dtype=float),
            "dist_next_curve": np.array([dist_next_curve if dist_next_curve is not None else -1.0], dtype=float),
            "angle_next_curve": np.array([curve_angle if curve_angle is not None else 0.0], dtype=float)
        }

        # print(obs)
        # self.history.append(obs)
        return obs


    def loss(self, data) -> float :
        
        position = data["position"]
        x = position[0]
        y = position[1]

        # Do something
        car_heading = data["direction"]
        
        dist = distance_to_end_on_racing_line([x, y], self.racing_line)
        d_dist = dist - self.prev_distance_to_end
        self.prev_distance_to_end = dist

        loss_value, d = line_ref_loss([x, y], car_heading, self.racing_line, k=1.0)

        self.distance_to_line = d
        if self.distance_to_line > 1 :
            return 100
        
        d_dist_term = d_dist * 20
        loss_value_term = float(loss_value) / 20
        print(f"line_term: {loss_value_term:.3f}, Dist to line: {d:.3f}, Dist to end: {dist:.3f}, d_dist term: {d_dist_term:.3f}")
        return d_dist_term + loss_value_term
        
