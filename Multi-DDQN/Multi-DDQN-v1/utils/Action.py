import numpy as np

class Actions(object):
    def __init__(self):
        # 对历史信息进行记录
        self.order = 0
        self.actions = np.zeros((66,4), dtype=bool)
        self.num = 66

    def get_actions(self):
        return self.actions

    def get_action(self, order):
        return self.actions[:,order]

    def set_action(self, action, order):
        if action.shape[0] != self.actions.shape[0]:
            raise ValueError("The number of rows in arrays action and actions must match!")
        # self.actions[:, order] = action
        for i in range(self.num):
            self.actions[i,order] = action[i]

    def get_order(self):
        return self.order

    def set_order(self, order):
        self.order = order
