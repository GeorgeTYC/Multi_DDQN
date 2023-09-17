import numpy as np

class VNFs:
    def __init__(self):
        self.vnf_cpu_consumption = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/vnf_cpu_consumption.npy')

    def vnf_cpu_consumption(self):
        return self.vnf_cpu_consumption

# vnf = VNFs()
# print(vnf.vnf_cpu_consumption)