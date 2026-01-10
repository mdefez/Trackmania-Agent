import sys
sys.path.append("..")

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
## Random model
from stable_baselines3 import SAC


# from Learning.agents.random.random_agent import RandomEnv
# env = RandomEnv()
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# print(model.predict(env.reset()[0]))

## Line Ref model
# from Learning.environments.ref_line_borderless.ref_line_baseline_env import RefFLineBaseLineEnv
from save import SaveEvery100StepsCallback
from Learning.environments.ref_line_poster.ref_line_baseline_env import RefFLineBaseLineEnv

env = RefFLineBaseLineEnv("./data/blocks.csv")

# import time 
# env.reset()
# while True :
#     obs = env._get_obs()
#     try : 
#         env._compute_reward(obs)
#     except Exception as e :
#         print("Error in reward computation:", e)
#     time.sleep(0.5)



save_callback = SaveEvery100StepsCallback(save_freq=4000, verbose=1)
# model = SAC.load("../trainings/sac_wall/model_step_15000", env = env)
# model.load_replay_buffer("../trainings/sac_wall/buffer_step_15000")
# 
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, callback = save_callback )


# noob.learn(total_timesteps=10000, callback = save_callback )

# ## Import Naive Model
# from Learning.models.naive import NaiveModel
# from Learning.environments.naive.naive_env import NaiveEnv

# env = NaiveEnv()
# model = NaiveModel(env)
# model.learn(total_timesteps=1000)

# print(model.predict(env.reset()[0]))