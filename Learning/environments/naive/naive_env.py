from setup.trackmania.env import TMEnv
import gymnasium as gym
from ...utils import new_features

class NaiveEnv(TMEnv) :
    def __init__(self):
        super().__init__(observation_space = gym.spaces.Dict({
            "speed": gym.spaces.Box(low=0, high=300, shape=(1,)),
            "finished": gym.spaces.Discrete(2),
            "time": gym.spaces.Box(low=0, high=1e6, shape=(1,)),
            "distance_next_turn": gym.spaces.Box(low=0, high=1e4, shape=(1,))
        }))
    
    def _action_to_key(self, action: gym.spaces.Space) -> list[str]:
        return(list(action))
    
    def _get_info(self):
        return {}
    
    def _is_truncated(self, obs):
        return False
    
    def _is_terminated(self, obs):
        return False

    def _compute_reward(self, obs) -> float:    
        speed = obs.get("speed", 0.0)
        finished = obs.get("finished", False)
        distance_next_turn = min(obs.get("distance_next_turn", 1e3), 1e3)

        r = speed - (distance_next_turn / 10)

        if finished:
            r += 1e5  # gros bonus de fin

        return r

    def _data_to_observation(self, data):

        # Build and add new features (make it a cool and packed function)
        position = data["vehicleData"]["position"]
        distance_next_turn = new_features.distance_to_next_turn(position)
        data["distance_next_turn"] = distance_next_turn

        # Make it a simple dict with one level of keys/values. At that point every values should be float. WIP
        data = new_features.keep_relevant_features(data)

