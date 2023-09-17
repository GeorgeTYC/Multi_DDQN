from .base.BaseRewards import BaseRewardParams, BaseRewards
from .VNFs import VNFs
from .Topology import Topology
from .State import State
import numpy as np
from heapq import heappop, heappush
from .Action import Actions


class Rewards(BaseRewards):
    cumulative_reward: float = 0.0

    def __init__(self, vnfs: VNFs):
        super.__init__()

        # vnf映射失败惩罚系数
        self.vnf_mapping_failure_penalty = 1.0
        # 链路连接失败惩罚系数
        self.connection_failure_penalty = 1.0
        # 链路映射成功奖励值
        self.connection_sucess_reward = 0.2

        self.max_bandwidth: float = 4800.0
        self.max_cpu: float = 8000.0

        self.vnf_cpu_consumption = vnfs.vnf_cpu_consumption
        self.vnfs_num_per_sfc = 4
        # TODO：？？？
        # State里面不包含剩余带宽信息，将拓扑信息引入
        # self.reset()

    def calculate_reward(self, state: State, action, topo: Topology):
        '''
                :param state: 初始状态
                :param action: 动作，将VNF放置在哪颗卫星节点之上
                :return:
                '''
        reward: float = -1.0

        # 考虑链路资源和带宽消耗
        rcost = 0.0
        nodes_cpu = state.nodes_cpu
        nodes_bandwidth = state.nodes_bandwidth

        # TODO：优化奖励函数
        # 判断是否映射成功
        if not self.ismapped(state, action):
            return reward

        # 映射成功，计算奖励
        # 节点资源和带宽的计算
        cpu_cost: float = 0.0
        bandwidth_cost: float = 0.0

        for i in range(4):
            node_mapping = action[:, i]
            # VNF类型
            vnf_type = state.sfc_request[:, i]
            if vnf_type.shape[0] != self.vnf_cpu_consumption.shape[0] or not vnf_type.shape[1] != \
                                                                             self.vnf_cpu_consumption.shape[1]:
                raise ValueError(
                    "The number of rows in arrays vnf_type and vnf_cpu_consumption must match")
            # VNF的CPU资源消耗
            cpu_consuption = (vnf_type * self.vnf_cpu_consumption).sum()
            if node_mapping.shape[0] != nodes_cpu.shape[0] or node_mapping.shape[1] != nodes_cpu.shape[1]:
                raise ValueError(
                    "The number of rows and cols in arrays node_mapping and nodes_cpu must match")
            # 指定节点的剩余CPU资源和带宽资源
            node_cpu = (node_mapping * nodes_cpu).sum()
            node_bandwidth = (node_mapping * nodes_bandwidth).sum()

            # 节点剩余资源小于资源需求，视为失败返回结果
            if cpu_consuption > node_cpu:
                return reward
            # 节点的带宽小于带宽需求，视为失败返回
            if node_bandwidth < state.bandwidth_request:
                return reward

            # 成本
            cpu_cost += (self.max_cpu / node_cpu)
            bandwidth_cost += (self.max_bandwidth / node_bandwidth)
            # 更新节点剩余CPU资源
            nodes_cpu = nodes_cpu - (cpu_consuption) * node_mapping

        # 链路的距离矩阵
        nodes_distance = state.nodes_distance
        max_distance = nodes_distance.max()
        nodes = np.hstack(state.ingress, action)
        nodes = np.hstack(nodes, state.egress)
        total_distance = self.get_total_distance(nodes, nodes_distance)
        delay_cost = total_distance / max_distance

        total_cost = cpu_cost + bandwidth_cost + delay_cost
        reward = (1 / total_cost)

        # 更新网络中的CPU资源

        state.nodes_cpu = nodes_cpu
        self.topology.nodes_cpu = nodes_cpu

        # 更新网络中的带宽资源
        remaining_bandwidth = self.topology.links_bandwidth
        remaining_bandwidth = self.update_remaining_bandwidth(remaining_bandwidth)
        self.topology.links_bandwidth = remaining_bandwidth

        state.nodes_bandwidth = state.calculate_bandwidth(remaining_bandwidth)
        state.nodes_distance, state.bandwidth_bottleneck = state.floyd_warshall_with_bandwidth(
            self.topology.links_distance, remaining_bandwidth)

        return reward, state

    def ismapped(self, state: State, action: np.ndarray):
        ## 判断是否映射成功
        # action： 每一个选择只有1个非零
        if action.shape[1] != 4:
            raise ValueError("The input array must have 4 columns")
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("The input array must contain integers")

        if not np.all(np.sum(action, axis=0) == 1):
            return False

        ## 判断是否满足限制条件和资源条件
        # m*4 矩阵
        sfc_request = state.sfc_request
        # m*1矩阵
        vnfs_matrix = self.vnf_cpu_consumption
        # 节点cpu资源，n*1
        nodes_cpu = state.nodes_cpu
        # 节点所支持的VNF类型，n*m
        vnfs_supported = state.type_vnf_matrix

        if sfc_request.shape[1] != 4 or action.shape[1] != 4:
            raise ValueError("Arrays sfc_request and action must have 4 columns")
        if nodes_cpu.shape[0] != sfc_request.shape[0] or vnfs_matrix.shape[0] != action.shape[0]:
            raise ValueError(
                "The number of rows in arrays nodes_cpu and sfc_request, and in arrays vnfs_matrix and action must match")
        if vnfs_supported.shape[0] != action.shape[0] or vnfs_supported.shape[1] != sfc_request.shape[0]:
            raise ValueError(
                "The number of rows in arrays vnfs_supported and action, and in arrays vnfs_supported and sfc_request must match")
        ## 判断限制条件是否满足
        result = np.dot(action, sfc_request.T) * vnfs_supported
        if not result.sum() == 4:
            return False

        ## 判断节点资源条件是否满足
        # 节点资源
        cpu_consumption = np.sum(sfc_request * vnfs_matrix, axis=0)
        cpu_remaining = np.sum(action * nodes_cpu, axis=0)
        if not np.all(cpu_consumption < cpu_remaining):
            return False

        # 带宽资源限制
        # 出入节点
        ingress_node = state.ingress
        egress_node = state.egress
        # 带宽限制
        min_bandwidth = state.bandwidth_bottleneck
        # 请求带宽
        bandwidth_request = state.bandwidth_request

        if ingress_node.shape[0] != action.shape[0] or action.shape[0] != egress_node.shape[0]:
            raise ValueError("The number of rows in arrays ingress_node, action and egress_node must match!")

        action_new = np.hstack(ingress_node, action, egress_node)
        for i in range(5):
            arr_former = action_new[:, i]
            arr_latter = action_new[:, i + 1]
            arr_temp = np.dot(arr_former, arr_latter.T)
            # 带宽不足
            if (arr_temp * min_bandwidth).sum() < bandwidth_request:
                return False

        return True

    def is_mapped_single_vnf(self, state: State, actions: Actions):
        ## 判断单VNF是否映射成功
        order = state.get_vnf_order()
        action = actions.get_action(order)
        # action: 一列只有1个非零
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("The input array must contain integers")

        if not np.sum(action, axis=0) == 1:
            return False

        ## 判断是否满足限制条件和资源条件
        # m*4 矩阵
        sfc_request = state.sfc_request
        # m*1矩阵
        vnfs_matrix = self.vnf_cpu_consumption
        # 节点cpu资源，n*1
        nodes_cpu = state.nodes_cpu
        # 节点所支持的VNF类型，n*m
        vnfs_supported = state.type_vnf_matrix

        if sfc_request.shape[1] != 4:
            raise ValueError("Arrays sfc_request must have 4 columns")
        if nodes_cpu.shape[0] != sfc_request.shape[0] or vnfs_matrix.shape[0] != action.shape[0]:
            raise ValueError(
                "The number of rows in arrays nodes_cpu and sfc_request, and in arrays vnfs_matrix and action must match")
        if vnfs_supported.shape[0] != action.shape[0] or vnfs_supported.shape[1] != sfc_request.shape[0]:
            raise ValueError(
                "The number of rows in arrays vnfs_supported and action, and in arrays vnfs_supported and sfc_request must match")

        ## 判断限制条件是否满足
        # 找到action数组中不为0的位置
        row_index = np.argmax(action)
        # 取出vnfs_supported数组中对应的行
        row = vnfs_supported[row_index]
        # 取出vnf
        vnf_request = sfc_request[:, order]
        result = np.dot(row, vnf_request)
        if not result == 1:
            return False

        ## 判断节点资源条件是否满足
        # 节点资源
        cpu_consumption = np.sum(np.dot(vnf_request, vnfs_matrix))
        cpu_remaining = np.sum(nodes_cpu[row_index])
        if not cpu_consumption < cpu_remaining:
            return False

        return True

    # 更新网络带宽资源
    def update_remaining_bandwidth(self, remaining_bandwidth: np.ndarray, links_distance: np.ndarray,
                                   nodes_list: np.ndarray, bandwidth_request):
        '''

        :param remaining_bandwidth: 链路剩余带宽，n*n
        :param links_distance: 链路长度，没有直接链接则为无穷大，n*n
        :param nodes_list: 出入节点卫星
        :param bandwidth_request: 带宽要求
        :return:
        '''
        n = nodes_list.shape[0]
        for i in range(n - 1):
            start = np.argmax(nodes_list[i])
            end = np.argmax(nodes_list[i + 1])
            path = self.dijkstra(links_distance, start, end)
            for j in range(len(path) - 1):
                head_node = path[i]
                tail_node = path[i + 1]
                remaining_bandwidth[head_node][tail_node] = max(0, remaining_bandwidth[head_node][
                    tail_node] - bandwidth_request)
                remaining_bandwidth[tail_node][head_node] = max(0, remaining_bandwidth[tail_node][
                    head_node] - bandwidth_request)

        return remaining_bandwidth

    # 计算最短路径
    def dijkstra(self, links_distance: np.ndarray, start, end):
        n = links_distance.shape[0]
        visited = np.zeros(n, dtype=bool)
        distances = np.full(n, np.inf)
        distances[start] = 0
        previous = np.full(n, -1, dtype=int)
        queue = [(0, start)]
        while queue:
            current_distance, current_node = heappop(queue)
            if visited[current_node]:
                continue
            visited[current_node] = True
            for i in range(n):
                distance = current_distance + links_distance[current_node][i]
                if distance < distances[i]:
                    distances[i] = distance
                    previous[i] = current_node
                    heappush(queue, (distance, i))
        path = []
        current_node = end
        while current_node != -1:
            path.append(current_node)
            current_node = previous[current_node]
        path.reverse()
        return path

    def get_total_distance(self, nodes: np.ndarray, nodes_distance: np.ndarray):
        nodes_num = nodes.shape[1]
        total_distance: float = 0.0
        for i in range(nodes_num - 1):
            node_pre = np.argmax(nodes[:, i])
            node_post = np.argmax(nodes[:, i + 1])
            total_distance += nodes_distance[node_pre, node_post]

        return total_distance

    def calculate_reward_V2(self, state: State, actions: Actions, topo: Topology):
        # TODO: 计算奖励
        # 获取VNF在sfc的顺序，最后一个和之前的reward计算方式不同
        order = state.get_vnf_order()
        if order < self.vnfs_num_per_sfc - 1:
            reward, state_new, topo_new = self.calculate_reward_single_vnf(state, actions, topo)
        else:
            # TODO: 完成最后的映射
            pass
        return reward, state_new, topo_new

    def calculate_reward_single_vnf(self, state: State, actions: Actions, topo: Topology):
        # 只计算单个vnf的映射情况，不考虑链路
        reward: float = self.vnf_mapping_failure_penalty

        # 仅考虑节点资源
        nodes_cpu = state.nodes_cpu
        nodes_bandwidth = state.nodes_bandwidth

        if not self.is_mapped_single_vnf(state, actions):
            return reward

        # 映射成功，计算奖励
        cpu_cost: float = 0.0
        bandwidth_cost: float = 0.0

        order = state.get_vnf_order()
        node_mapping = actions.get_action(order)
        # VNF类型
        vnf_type = state.sfc_request[:, order]
        if vnf_type.shape[0] != self.vnf_cpu_consumption.shape[0]:
            raise ValueError(
                "The number of rows in arrays vnf_type and vnf_cpu_consumption must match")

        # VNF的CPU资源消耗
        cpu_consuption = np.sum(np.dot(vnf_type, self.vnf_cpu_consumption))
        if node_mapping.shape[0] != nodes_cpu.shape[0]:
            raise ValueError(
                "The number of rows and cols in arrays node_mapping and nodes_cpu must match")
        # 指定节点的剩余CPU资源和带宽资源
        node_cpu = np.sum(np.dot(node_mapping, nodes_cpu))
        node_bandwidth = np.sum(np.dot(node_mapping, nodes_bandwidth))

        if cpu_consuption > node_cpu:
            return reward
        # 节点的带宽小于带宽需求，视为失败返回
        if node_bandwidth < state.bandwidth_request:
            return reward

        cpu_cost = (self.max_cpu / node_cpu)
        bandwidth_cost = (self.max_bandwidth / node_bandwidth)
        row_index = np.argmax(node_mapping)
        nodes_cpu[row_index] = node_cpu - cpu_consuption

        total_cost = cpu_cost + bandwidth_cost

        reward = (1 / total_cost)

        # 更新网络中的CPU资源
        state.nodes_cpu = nodes_cpu
        topo.nodes_cpu = nodes_cpu

        return reward, state, topo

    def calculate_reward_single_vnf_with_links(self, state: State, actions: Actions, topo: Topology):
        # 计算最后一个节点和链路的映射情况
        reward_vnf: float = self.vnf_mapping_failure_penalty
        reward_link: float = self.connection_failure_penalty

        reward_vnf, state_new, topo_new = self.calculate_reward_single_vnf(state=state, actions=actions, topo=topo)

    def calculate_reward_links(self, state: State, actions: Actions):
        # 计算映射链路的奖励
        reward_link: float = self.connection_failure_penalty

        # 带宽资源限制
        # 出入节点
        ingress_node = state.ingress
        egress_node = state.egress
        # 带宽限制
        min_bandwidth = state.bandwidth_bottleneck
        # 请求带宽
        bandwidth_request = state.bandwidth_request

        # 动作集合，n*4
        action = actions.get_actions()

        if ingress_node.shape[0] != action.shape[0] or action.shape[0] != egress_node.shape[0]:
            raise ValueError("The number of rows in arrays ingress_node, action and egress_node must match!")

        action_new = np.hstack(ingress_node, action, egress_node)

        reward_temp: float = 0.0

        for i in range(self.vnfs_num_per_sfc + 1):
            arr_former = action_new[:, i]
            arr_latter = action_new[:, i + 1]
            row_index = np.argmax(arr_former)
            col_index = np.argmax(arr_latter)
            # 带宽充足
            if (row_index == col_index) or (min_bandwidth[row_index, col_index] >= bandwidth_request):
                reward_temp += self.connection_sucess_reward





