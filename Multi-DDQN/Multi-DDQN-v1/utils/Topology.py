from skimage import io
import numpy as np

'''
    获取拓扑的相关信息
'''
class Topology:
    def __init__(self, nodesIds, links, remaining_cpu, remaining_bandwidth, links_distance, vnfs_supported, nodes_loc):
        # 卫星节点编号数据
        self.nodesIds = nodesIds
        # 链路连接信息，0-1表示
        self.links = links
        # 设置节点的剩余CPU资源，单位MIPS
        self.nodes_cpu = remaining_cpu
        # 设置带宽资源，n*n
        self.links_bandwidth = remaining_bandwidth
        # 链路长度 n*n
        self.links_distance = links_distance
        # 节点所能支持的VNF类型 n*m
        self.vnfs_supported = vnfs_supported

        # 节点位置
        self.nodes_loc = nodes_loc


    def get_nodesIds(self):
        return self.nodesIds

    def get_links(self):
        return self.links

    def get_cpu(self):
        return self.nodes_cpu

    def set_nodes_cpu(self, nodes_cpu):
        self.nodes_cpu = nodes_cpu

    def get_bandwidth(self):
        return self.links_bandwidth

    def set_links_bandwidth(self, links_bandwidth):
        self.links_bandwidth = links_bandwidth

    def get_links_distance(self):
        return self.links_distance

    def set_links_distance(self, links_distance):
        self.links_distance = links_distance

    def get_vnfs_supported(self):
        return self.vnfs_supported

    def set_vnfs_supported(self, vnfs_supported):
        self.vnfs_supported = vnfs_supported


def load_topology(path, remaining_cpu, remaining_bandwidth, vnfs_supported):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return Topology(data, remaining_cpu, remaining_bandwidth, vnfs_supported)

