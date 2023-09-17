
class BaseRewards:
    def __init__(self):
        self.cumulative_reward: float = 0.0

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def reset(self):
        self.cumulative_reward = 0
