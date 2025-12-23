import gymnasium as gym
from flask import Flask, request, jsonify

import asyncio
import threading

import platform
if platform.system() == "Darwin":
    from .keyboard.macos import PressKey, ReleaseKey
elif platform.system() == "Windows":
    from .keyboard.windows import PressKey, ReleaseKey
from .keyboard.generic import W, A, S, D

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

from ..config import WINDOWS_IP
import requests

import time

class TMController:
    def __init__(self, app : Flask):

        ## HTTTP server
        self.server = Flask("TMServer")
        ## Async utils
        self._loop = asyncio.get_event_loop()
        self._data_future = self._loop.create_future()

        # Keys
        self.new_press_keys = set()

        @self.server.route("/api/data", methods=["POST"])
        def handle_data():
            data = request.get_json()

            if not data :
                return jsonify({"error": "No JSON payload provided"}), 400
            
            def set_data(data) :
                if not self._data_future.done() :
                    self._data_future.set_result(data)

            self._loop.call_soon_threadsafe(set_data, data)

            return jsonify({"status": "success", "received": data}), 200
    
    def start(self):
        self.thread = threading.Thread(target= self.server.run, kwargs={"host":"0.0.0.0", "port":8080})
        self.thread.daemon = True
        self.thread.start()
    
    def _reset_future(self):
        self._data_future = self._loop.create_future()

    def _wait_data(self):
        data = self._loop.run_until_complete(self._data_future)
        self._reset_future()
        return data

    def action(self, keys : list[str]):
        ## Clear all pressed keys
        old_pressed_keys = self.new_press_keys.copy()
        self.new_press_keys.clear()

        # Press and Release keys accordingly
        for key in keys:
            # print("Key to press:", key)
            self.new_press_keys.add(key)

        payload = {
            "press": [],
            "release": []
        }
        for k in self.new_press_keys - old_pressed_keys:
            payload["press"].append(k)
            # print("Pressing key:", k)
            # PressKey(k)
        for k in old_pressed_keys - self.new_press_keys:
            payload["release"].append(k)
            # print("Releasing key:", k)
            # ReleaseKey(k)
        # print(payload)
        requests.post(f"http://{WINDOWS_IP}:8081/command", json=payload)
    
    def reset(self):
        payload = {
            "press": ["Backspace"],
            "release": ["Backspace"]
        }
        while True :
            # Test and retry if timeoue
            try :
                requests.post(f"http://{WINDOWS_IP}:8081/reset", json=payload, timeout=1)
                break
            except requests.exceptions.RequestException as a :
                print("Timeout while resetting controller, retrying...")
                continue


        print("Resetting controller...")


class TMEnv(gym.Env):
    def __init__(self, observation_space: gym.spaces.Space):
        self.controller = TMController(app=None)
        self.controller.start()

        # Spaces
        # [steering, throttle, brake]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.observation_space = observation_space

    def _action_to_key(self, action: gym.spaces.Space) -> list[str]:
        """ From a gym.spaces.Box((3, )), returns a list of keys to press """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _compute_reward(self, obs) -> float:
        """ From a data retrieved from the server, compute the reward """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _is_terminated(self, obs) -> bool:
        """ Tells if the episode can be considered as terminated """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _is_truncated(self, obs) -> bool:
        """ Tells if the episode can be considered as truncated (ie time limit reached) """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _data_to_observation(self, raw_data) -> gym.spaces.Space:
        """
        From a data retrieved from the server, compute the observation and map it to observation_space
        Note that any other method called with a data paremeter will receive the observation living in observation_space 
        as input, not the raw data.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _get_obs(self) :
        data = self.controller._wait_data()
        return self._data_to_observation(data)
    
    def _get_info(self):
        return {}
    
    def reset(self, seed = None, options = None):
        self.controller.reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        ## Action
        keys = self._action_to_key(action)
        self.controller.action(keys)

        ## State
        obs = self._get_obs()
        rew = self._compute_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = self._is_truncated(obs)
        info = self._get_info()

        return obs, rew, terminated, truncated, info

    
