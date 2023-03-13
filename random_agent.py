import numpy as np


class RandomAgent(object):
    def __init__(self,
                 act_n):
        self.act_n = act_n

    def predict(self, obs, avail_actions):
        action = np.random.choice(np.where(avail_actions == 1)[0])
        return action
