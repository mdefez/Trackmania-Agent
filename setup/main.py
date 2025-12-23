import sys
sys.path.append("..")

## Random model
# from stable_baselines3 import SAC
# from Learning.agents.random.random_agent import RandomEnv

# env = RandomEnv()
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# print(model.predict(env.reset()[0]))

## Line Ref model
from stable_baselines3 import SAC
from Learning.environments.ref_line_tm.ref_line_agent_continuous import RefLineEnvContinuous

env = RefLineEnvContinuous("./data/blocks.csv")

import time 
# env.reset()
# while True :
#     obs = env._get_obs()
#     try : 
#         env._compute_reward(obs)
#     except Exception as e :
#         print("Error in reward computation:", e)
#     time.sleep(1)


model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


# ## Import Naive Model
# from Learning.models.naive import NaiveModel
# from Learning.environments.naive.naive_env import NaiveEnv

# env = NaiveEnv()
# model = NaiveModel(env)
# model.learn(total_timesteps=1000)

# print(model.predict(env.reset()[0]))