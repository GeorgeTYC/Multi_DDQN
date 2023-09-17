import numpy as np
from collections import deque
from utils.State import State
from utils.VNFs import VNFs
from utils.Rewards import Rewards
from utils import matReader
from utils.Topology import Topology
from utils.Action import Actions
from utils.Utils import Utils
from utils.ServiceBuffer import ServiceBuffer

class Enviroment(object):
    def __init__(self):
        self.state_buffer = deque()
        # 初始CPU单位，MIPS
        self.init_cpu = 8000.0
        # 初始带宽单位，MBps
        self.init_bandwidth = 1200.0
        # 每种vnf类型的cpu消耗
        self.vnfs = VNFs()
        self.rewards = Rewards(self.vnfs)
        self.topology = self.get_topology()
        self.sfc_length = 4
        self.training_steps = 0

        self.pos_converter = Utils()
        self.serviceBuffer = ServiceBuffer()

        # TODO： 修改step为1000
        self.total_steps = 4

    def step(self, actions: Actions, state: State):
        # 获取奖励，新的状态以及是否终止的信息
        reward, state_new, topo = self.rewards.calculate_reward_V2(state=state, actions=actions, topo=self.topology)
        # 更新拓扑
        self.topology = topo

        order = state.get_vnf_order()
        # vnf未到达最后一个，state进入下一个状态更新vnf
        if order < self.sfc_length - 1:
            state_new.next_vnf(actions)
        else:
            ## 最后一个vnf执行完毕，state开始调用执行下一条sfc
            # 获取下一条sfc
            vnfs, bandwidth, source_pos, destination_pos = self.serviceBuffer.next_sfc()
            # 将经纬度转换为接入、接出节点
            source_satellite = self.pos_converter.get_nearest_node(self.topology, source_pos)
            destination_satellite = self.pos_converter.get_nearest_node(self.topology, destination_pos)
            # state进入下一个状态
            state_new.next_sfc(actions, vnfs, bandwidth, source_satellite, destination_satellite)

        self.training_steps += 1

        done = False
        if self.training_steps == self.total_steps:
            done = True
            self.training_steps = 0

        return state_new, reward, done

    def reset(self, e):
        ## 参数e为episode
        # 环境重置
        self.steps = 0
        self.state_buffer = deque()
        self.topology = self.get_topology()
        self.rewards = Rewards(self.vnfs)

        # 获取指定的SFC
        vnfs, bandwidth, source_pos, destination_pos = self.serviceBuffer.get_sfc(e)
        # 将经纬度转换为接入、接出节点
        source_satellite = self.pos_converter.get_nearest_node(self.topology, source_pos)
        destination_satellite = self.pos_converter.get_nearest_node(self.topology, destination_pos)

        return State(self.topology.nodes_cpu, self.topology.links_bandwidth, self.topology.links_distance, vnfs, 0, bandwidth, source_satellite, destination_satellite)

    def get_topology(self, time=0):
        matConveter = matReader.matReader(time)
        nodesIds, links, links_distance, nodes_loc = matConveter.get_topology_info()
        # 获取每颗卫星支持的vnfs编号，尺寸为n*m
        vnfs_supported = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/vnfs_supported.npy')
        num_sate = vnfs_supported.shape[0]
        init_cpu = np.full((num_sate,1), self.init_cpu)
        init_bandwidth = links * self.init_bandwidth
        return Topology(nodesIds=nodesIds, links=links, remaining_cpu=init_cpu, remaining_bandwidth=init_bandwidth, links_distance=links_distance, vnfs_supported=vnfs_supported, nodes_loc=nodes_loc)

    def get_state_dim(self):
        remaining_cpu = self.topology.nodes_cpu
        remaining_bandwidth = self.topology.links_bandwidth
        distance = self.topology.links_distance
        vnfs, bandwidth, source_pos, destination_pos = self.serviceBuffer.get_example()
        order = 0
        ingress = self.pos_converter.get_nearest_node(self.topology, source_pos)
        egress = self.pos_converter.get_nearest_node(self.topology, destination_pos)
        state_example = State(remaining_cpu, remaining_bandwidth, distance, np.zeros((8,4)), order, bandwidth, ingress, egress)
        return state_example.reshape_state().shape[0]

    def get_action_dim(self):
        return 66