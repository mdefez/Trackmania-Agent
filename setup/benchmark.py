
import sys
sys.path.append("..")

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from Learning.environments.ref_line_poster.ref_line_baseline_env import RefFLineBaseLineEnv

env = RefFLineBaseLineEnv("./data/blocks.csv", benchmark_mode=True)

import numpy as np

import time

N_TEST = 20

DUMP_PATH = "../trainings/test_bench/"



def test_single_model(model_path):
    model = SAC.load(model_path, env=env)

    lap_times = []

    for episode in range(N_TEST):
        env.reset()
        time.sleep(1)
        obs, _ = env.reset()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic = False)
            obs, reward, done, _, _ = env.step(action)
        
        if env.finished :
            print(f"Finished in : {env.finished_in}")
            lap_times.append(env.finished_in)
        else : 
            print("Did not finish")
            lap_times.append(np.inf)
    
    return lap_times


# For each zip in the MODEL_PATH directory, run N_TEST episodes and save the rewards in a csv file
import os
from stable_baselines3 import SAC
from collections import defaultdict

step_result = defaultdict(list)

for file in os.listdir(os.path.dirname(DUMP_PATH)):
    if file.endswith(".zip"):
        model_path = os.path.join(os.path.dirname(DUMP_PATH), file)
        steps = file.split("_")[-1].split(".")[0]
        print(f"Testing model at step {steps}")
        lap_times = test_single_model(model_path)
        step_result[steps] = lap_times

# Dump results to csv
import pandas as pd
df = pd.DataFrame.from_dict(step_result, orient='index')
df.to_csv(os.path.join(DUMP_PATH, "benchmark_results.csv"))

print(step_result)