keys = ["W", "A", "S", "D"]
possible_actions = {(1, 1) : "WD", (1, -1) : "WA", (1, 0) : "W", (0, -1) : "A", (0, 1) : "D"}

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def compute_discounted_reward(
    rewards,
    gamma=0.99
):
    """
    rewards : liste de rewards successives [r_t, r_{t+1}, ..., r_{t+k}]
    gamma  : facteur de décroissance (0 < gamma <= 1)

    retourne : reward scalaire
    """

    total_reward = 0.0
    power = 0

    for reward in rewards:
        total_reward += (gamma ** power) * reward
        power += 1

    return total_reward

from setup.trackmania.env import TMEnv

class NaiveModel :
    def __init__(self, env : TMEnv):
        self.env = env
        self.history = [] 
        self.k = 10  # Calcul de la reward sur les k prochains états
        self.gamma = 0.99

        self.history_action = []    # Garde en mémoire les actions effectuées
        self.history_reward = []

        # Init network
        self.neural_network = RewardNetwork(state_dim=len(self.env.observation_space), action_dim=self.env.action_space.shape[0], hidden_dim=128)

    def learn(self, total_timesteps):
        first_obs = self.env.reset()
        self.history.append(first_obs)
        for t in range(total_timesteps):
            ## Compute action
            next_action = self.find_action()

            ## Apply action
            obs, reward, terminated, truncated, info = self.env.step(next_action)

            self.history_reward.append(reward)
            self.history.append(obs)
            self.update_weights()
    
    def find_action(self):
        max_expected_reward = -float("inf")
        current_data = list(self.history[-1].values())

        for action in possible_actions.keys():
            expected_reward = self.neural_network.forward(torch.tensor(current_data, dtype = torch.float32), torch.tensor(action, dtype = torch.float32))
            if expected_reward > max_expected_reward:
                best_action = action 

        # epsilon greedy pour explorer un peu
        A = random.random()
        eps = 0.2
        if A < 1-eps:
            choosen_action = best_action
        
        else:
            choosen_action=  random.choice(list(possible_actions.keys()))

        self.history_action.append(choosen_action)
        return possible_actions[choosen_action]
    
    def update_weights(self):
        """
        Fonction qui appelle la fonction de mise à jour des poids. Elle est effectué à chaque étape uniquement après l'étape k (pour pouvoir calculer les vraies loss)
        """

        # Dès qu'on a k états futurs, on entraîne
        if len(self.history) > self.k:
            
            idx = -self.k

            future_states = [h for h in self.history[idx:idx+self.k - 1]]
            future_states.append(self.history[-1])

            true_reward = compute_discounted_reward(future_states, self.gamma)
            true_reward = torch.tensor(true_reward, dtype=torch.float32).unsqueeze(0)

            state_t = torch.tensor(list(self.history[idx].values()), dtype = torch.float32)
            action_t = torch.tensor(self.history_action[idx], dtype = torch.float32)

            self.neural_network.update_weights(
                state_t, action_t,
                true_reward
            )


# Prend en entrée les données et une action à évaluer. Renvoie une estimation de la reward si on choisit cette action
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.lr = 1e-4

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state, action):
        """
        state  : tensor (state_dim)
        action : tensor (action_dim)
        """
        
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x)
    
    def update_weights(self, state, action, R_true):
        """
        state, action : tenseur de l'instant t. On les utilise pour calculer la prédiciton de la discounted reward (faite à l'instant t)
        R_true : tenseur scalaire (reward vraie calculée à t+k)
        """
        R_pred = self.forward(state, action)

        loss = nn.MSELoss()(R_pred, R_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss.item()