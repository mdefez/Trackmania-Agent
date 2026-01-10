from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os

BASE_PATH = "../trainings/sac_lookahead_wall_times_boosted/"

class SaveEvery100StepsCallback(BaseCallback):
    def __init__(self, save_freq=100, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        # Save every N steps
        if steps % self.save_freq == 0:
            model_path = os.path.join(BASE_PATH, f"model_step_{steps}")
            buffer_path = os.path.join(BASE_PATH, f"buffer_step_{steps}")

            self.model.save(model_path)
            self.model.save_replay_buffer(buffer_path)

            raw_env = self.training_env.envs[0].unwrapped
            
            perfs = raw_env.perfs
            perfs_path = os.path.join(BASE_PATH, f"perfs_step_{steps}.txt")
            with open(perfs_path, "w") as f:
                for perf in perfs:
                    f.write(f"{perf}\n")

            raw_env.perfs = []

            if self.verbose:
                print(f"Saved model & buffer at step {steps}")

        # Optional: save on env signal
        # for info in self.locals.get("infos", []):
        #     if info.get("save_now"):
        #         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #         self.model.save(f"{BASE_PATH}/model_event_{timestamp}")
        #         self.model.save_replay_buffer(f"{BASE_PATH}/buffer_event_{timestamp}")

        return True
