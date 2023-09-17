import numpy as np

class ServiceBuffer:

    def __init__(self):
        # 服务功能链的vnfs请求，shape=[10000, 8, 4]
        self.requests_vnfs = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/requests_vnfs.npy')
        # SFC的带宽请求，shape=[10000, 1]
        self.requests_bandwidth = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/requests_bandwidth.npy')
        # 请求的经纬度，shape=[10000, 2]
        self.requests_source = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/requests_source.npy')
        self.requests_destination = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/requests_destination.npy')

        self.sfc_num = 0
        self.total_num = 10000

    def next_sfc(self):
        order = (self.sfc_num) % (self.total_num)
        vnfs = self.requests_vnfs[order,:]
        bandwidth = self.requests_bandwidth[order]
        source = self.requests_source[order,:]
        destination = self.requests_destination[order,:]

        self.sfc_num += 1
        return vnfs, bandwidth, source, destination

    def get_example(self):
        vnfs = self.requests_vnfs[0, :]
        bandwidth = self.requests_bandwidth[0]
        source = self.requests_source[0, :]
        destination = self.requests_destination[0, :]

        return vnfs, bandwidth, source, destination

    def get_sfc(self, order):
        vnfs = self.requests_vnfs[order, :]
        bandwidth = self.requests_bandwidth[order]
        source = self.requests_source[order, :]
        destination = self.requests_destination[order, :]

        self.sfc_num = order + 1
        return vnfs, bandwidth, source, destination





