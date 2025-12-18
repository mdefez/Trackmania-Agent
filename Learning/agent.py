keys = ["W", "A", "S", "D"]

import random

import pandas as pd
from rewards.ref_line import line_ref_loss

class Agent :
    def __init__(self):
        self.history = []
        
        df = pd.read_csv("clean_blocks.csv", sep=';')
        self.racing_line = df[['X','Z']].to_numpy()

    def feed(self, data) -> list[str]:
        self.history.append(data)
        return random.sample(keys, k=random.randint(0, len(keys)))

    def loss(self, data) -> float :
        vdata = data["vehicleData"]
        position = vdata["position"]
        x = position[0]
        y = position[1]

        # Do something
        car_heading = vdata["direction"]
        
        loss_value = line_ref_loss([x, y], car_heading, self.racing_line, k=1.0)
        return loss_value
        
