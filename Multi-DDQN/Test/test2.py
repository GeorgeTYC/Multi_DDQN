from Test.test1 import test1
import copy

class test2:
    def __init__(self):
        return

    def cal(self, test1: test1):
        testTemp = copy.deepcopy(test1)
        testTemp.set_order(2)
        return testTemp