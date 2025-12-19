from setup.trackmania.env import TMEnv

keys = ["W", "A", "S", "D"]

import random
import gymnasium as gym

import pandas as pd
from .ref_line import line_ref_loss


class RefLineEnv(TMEnv) :
    def __init__(self, track_path_csv: str) :
        super().__init__(observation_space=gym.spaces.Dict({
            "position": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,)),
            "direction": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=()),
        }))
        self.history = []
        self.track_path = track_path_csv

        df = pd.read_csv(self.track_path, sep=';')
        self.racing_line = df[['X','Z']].to_numpy()

    def _action_to_key(self, action):
        print(action)
        return random.sample(keys, k=random.randint(0, len(keys)))
    
    def _compute_reward(self, data):
        reward = - self.loss(data)
        print(reward)
        return reward

    def _is_terminated(self, data):
        return False
    
    def _is_truncated(self, data):
        return False
    
    def _data_to_observation(self, data):
        vdata = data["vehicleData"]
        self.history.append(vdata)
        return vdata

    def loss(self, data) -> float :
        
        position = data["position"]
        x = position[0]
        y = position[1]

        # Do something
        car_heading = data["direction"]
        
        loss_value = line_ref_loss([x, y], car_heading, self.racing_line, k=1.0)
        return loss_value
        
