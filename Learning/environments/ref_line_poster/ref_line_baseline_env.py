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

MAX_DIST_TO_LINE = 10.2
MAX_ELAPSED_TIME = 60.0 * 1000

HISTORY_LENGTH = 20
SPEED_HISTORY_THRESHOLD = 1
STEP_TIME_DEBUFF = 5

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

class RefFLineBaseLineEnv(TMEnv) :
    def __init__(self, track_path_csv: str, benchmark_mode: bool = False):
        super().__init__(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32))
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

        self.perfs = []

        self.track = TrackmaniaTrack(data)
        self.track.compute_pivots()

        self.need_save = False
        self.benchmark_mode = benchmark_mode
    
    def _action_to_key(self, action):
        # print(action)
        keys = []
        if action[0] > 0.1 :
            keys.append("W")
        if action[1] < -0.1 :
            keys.append("A")
        if action[1] > 0.1 :
            keys.append("D")
        # if action[2] > 0.1 :
        #     keys.append("S")
        return keys
    
    def _compute_reward(self, data):
        reward = self.reward(data)
        if not self.benchmark_mode :
            print("reward :", reward)
        return reward
    
    def reset(self, seed=None, options=None):
        obs, _info = super().reset(seed, options)
        self.start_y = 685
        self.start_time = None
        time.sleep(1.5)

        self.elapsed_time = 0
        self.prev_distance_to_end = None
        self.speed_history = []

        self.finished = False
        return obs, _info
    
    def _get_info(self):
        if self.need_save:
            self.need_save = False
            return {"save_now": True}
        return {}

    def _is_terminated(self, data):

        if self.benchmark_mode :
            if self.track.is_finished([self.x, self.y]) :
                self.finished = True
                self.finished_in = self.elapsed_time - 1000
                print("Finished line in benchmark mode", self.start_time, self.elapsed_time)
                return True
            
            if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
                print("Terminated: time out in benchmark mode")
                return True

            if self.y < self.start_y:
                return True

            else :
                return False


        # if abs(self.distance_to_median_line) > MAX_DIST_TO_LINE :
        #     print("Terminated: too far from line")
        #     return True

        if self.y < self.start_y:
            print("Terminated: went backwards")
            self.perfs.append(-1)
            return True
        # if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
        #     print("Terminated: time out")
        #     return True

        if self.track.is_finished([self.x, self.y]) :
            self.need_save = True
            self.perfs.append(self.elapsed_time)
            print("Terminated: finished line", self.start_time, self.elapsed_time)
            return True

        if len(self.speed_history) >= HISTORY_LENGTH and max(self.speed_history) < SPEED_HISTORY_THRESHOLD :
            self.perfs.append(-1)
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
        try : 
            line_angle = self.track.compute_line_angle(position)
            line_dist = self.track.distance_to_median_line(position)
            # line_dist = 0 if abs(line_dist) < 1e-6 else abs(line_dist)
            next_curve_dist = self.track.distance_to_next_curve(position)
            next_curve_angle = self.track._next_curve_angle(position)
        except :
            line_angle = 0
            line_dist = 0
            next_curve_dist = 200
            next_curve_angle = 0

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
        try : 
            angle_to_median_line = self.track.angle_to_line(car_heading_vec, position)
        except :
            angle_to_median_line = 0

        # print("Car heading vec : ", car_heading_vec)
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

        ## COmpute max speed in last 3 observation
        self.speed = min(max(self.speed, 0), 100)
        self.speed_history = getattr(self, 'speed_history', [])
        self.speed_history.append(self.speed)
        if len(self.speed_history) > HISTORY_LENGTH :
            self.speed_history.pop(0)

        debug_data = False
        if debug_data :
            # print("Position:", position)
            print("Direction:", direction)
            print("Dist to next curve:", dist_next_curve)
            print("Angle to next curve:", angle_next_curve)
            print("Distance to median line:", distance_to_median_line)
            print("Angle to median line:", angle_to_median_line)
            print("Car heading angle:", car_heading_angle)
            print("Speed:", speed)


        obs_vector = np.concatenate([
            # position, # 2
            # direction, # 3
            # dist_next_curve / 200, # 1
            # angle_next_curve, # 1
            dist_next_curve,
            distance_to_median_line / 10, # 1
            angle_to_median_line, # 1
            # car_heading_angle, # 1,
            speed
        ], axis=0)


        if not self.benchmark_mode :
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

        wall_term = 0
        if distance_to_line > MAX_DIST_TO_LINE :
            wall_term = -50
        #     print((self.speed_history))
        #     return  -(abs(self.angle_to_median_line) * max(self.speed_history) - self.track.compute_distance_to_finish(position) / 10) / 10
        #     # return (400 - self.track.compute_distance_to_finish(position)) * 2

        if self.y < self.start_y:
            print("Went backwards")
            print("y:", self.y, "start_y:", self.start_y)
            return -300 / 100
        
        # if self.elapsed_time and self.elapsed_time > MAX_ELAPSED_TIME :
        #     print("Terminated: time out")
        #     return True

        # if self.track.is_finished([self.x, self.y]) :
        #     # self.need_save = True
        #     return 5000 / (self.elapsed_time / 1000)
        
        if len(self.speed_history) >= HISTORY_LENGTH and max(self.speed_history) < SPEED_HISTORY_THRESHOLD :
            print("Stuck")
            return -100 / 100
        
        ## Normaliation and combination
        d_dist_term = - d_dist * 10
        dist_to_finish_term = - self.track.compute_distance_to_finish(position)

        # d_dist_term = min(max(d_dist_term, -1e3), 1e3)

        self.speed = min(max(self.speed, 0), 100)
        speed_term = self.speed * 5

        # dist_to_line_term = - float(abs(self.distance_to_median_line) * 3)
        # dist_to_line_term = 0
        # angle_to_line_term = - float(abs(median_line_angle) * 5) ** 2
        line_proximity = self.distance_to_median_line * median_line_angle * (1 if median_line_angle < 0 else -1)

        # line_procimity_term = line_proximity * 20 * 0
        angle_to_line_term = - (abs(self.angle_to_median_line) * 10) ** 2
        dist_to_line_term =- ( abs(self.distance_to_median_line) * 2 ) ** 2

        time_debuff = - STEP_TIME_DEBUFF
        # time_debuff = - self.elapsed_time / 700
        

        _cooked_feature_term = - (self.distance_to_median_line * self.angle_to_median_line * 2) ** 2

        paper_term = (self.speed + 2) * (np.cos(median_line_angle) - abs(distance_to_line) / 15) * 10
        # paper_term = self.speed ** 2 * (np.cos(median_line_angle) - abs(distance_to_line) / 10)

        if not self.benchmark_mode :
            print("time_debuff:", time_debuff)
            print("Line angle", median_line_angle)
            print(f"paper_term: {paper_term}, dist_term: {dist_to_line_term:.3f}, angle_term: {angle_to_line_term:.3f}, d_dist term: {d_dist_term:.3f}, speed_term: {speed_term:.3f}, dist_to_line_term: {dist_to_line_term:.3f}")

        f_sign = 1 if d_dist_term >= 0 else -1
        big_angle_penalisation = (self.angle_to_median_line * 2 ) ** 2

        # if d_dist_term < 0:
        #    paper_term = - abs(paper_term)

        return (paper_term + wall_term + time_debuff ) / 100
        # return paper_term + d_dist_term + wall_term + time_debuff