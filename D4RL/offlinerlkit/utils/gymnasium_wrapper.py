import gymnasium

class GymnasiumBoxWrapper(gymnasium.spaces.Box):
    def __init__(self, gym_space):
        super().__init__(gym_space.low, gym_space.high, gym_space.shape, gym_space.dtype)