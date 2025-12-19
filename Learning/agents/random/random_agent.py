from setup.trackmania.env import TMEnv
import gymnasium as gym
import numpy as np
import random

keys = ["W", "A", "S", "D"]

class RandomEnv(TMEnv):
    def __init__(self):
        observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        super().__init__(observation_space)

    def _action_to_key(self, action: gym.spaces.Space) -> list[str]:
        return random.sample(keys, k=random.randint(0, len(keys)))
    
    def _compute_reward(self, data) -> float:
        return 0
    
    def _is_terminated(self, data) -> bool:
        return False

    def _is_truncated(self, data) -> bool:
        return False
    
    def _data_to_observation(self, data) -> gym.spaces.Space:
        return np.array([data['test']])