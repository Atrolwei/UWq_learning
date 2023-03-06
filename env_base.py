class Env:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def step(self,action):
        raise NotImplementedError