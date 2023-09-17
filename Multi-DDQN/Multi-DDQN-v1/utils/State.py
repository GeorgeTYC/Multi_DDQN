import numpy as np
from utils.Action import Actions


class State:
    def __init__(self, remaining_cpu, remaining_bandwidth, distance, sfc_request, order, bandwidth_request, ingress, egress):
        '''

        :param nodes_cpu: 星座节点信息-剩余CPU资源 n*1
        :param remaining_bandwidth: 卫星链路通断-链路信息（链路剩余带宽） n*n
        :param type_vnf_matrix: 每颗卫星能够支撑的VNF类型 n*m
        :param sfc_request: SFC的VNF请求信息，即每个VNF的种类 m*4
        :param ingress: 请求的入节点 n*1
        :param egress: 请求的出节点 n*1
        '''
        # 节点剩余资源
        self.nodes_cpu = remaining_cpu
        # 节点的出入带宽
        self.nodes_bandwidth = self.calculate_bandwidth(remaining_bandwidth)
        # 节点之间的最短距离和带宽瓶颈
        self.nodes_distance, self.bandwidth_bottleneck = self.floyd_warshall_with_bandwidth(distance, remaining_bandwidth)
        # 节点能够支持的VNF矩阵，n*m
        self.type_vnf_matrix = np.load('E:/projects/Multi-DDQN/Multi-DDQN-v1/data/vnfs_supported.npy')
        # SFC的VNF请求信息，即每个VNF的种类 m*4
        self.sfc_request = sfc_request
        # 需要映射的VNF顺序
        self.vnf_order = order
        # 请求带宽
        self.bandwidth_request = bandwidth_request
        # 出入节点
        self.ingress = ingress
        self.egress = egress


    def calculate_bandwidth(self, remaining_bandwidth: np.ndarray):
        return remaining_bandwidth.sum(axis=1)

    def floyd_warshall(self, graph:np.ndarray):
        nodes_distance = graph
        n = graph.shape[0]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    nodes_distance[i,j] = min(nodes_distance[i,j], nodes_distance[i,k]+nodes_distance[k,j])
        return nodes_distance

    # 计算节点之间的最短链路和带宽瓶颈
    def floyd_warshall_with_bandwidth(self, distance: np.ndarray, reamining_bandwidth: np.ndarray):
        n = distance.shape[0]
        dist = np.array(distance)
        min_bandwidth = np.array(reamining_bandwidth)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        min_bandwidth[i][j] = min(min_bandwidth[i][k], min_bandwidth[k][j])
                    elif dist[i][k] + dist[k][j] == dist[i][j]:
                        min_bandwidth[i][j] = max(min_bandwidth[i][j], min(min_bandwidth[i][k], min_bandwidth[k][j]))
        return dist, min_bandwidth

    def get_type_vnf_matrix(self):
        return self.type_vnf_matrix

    def set_type_vnf_matrix(self, type_vnf_matrix):
        self.type_vnf_matrix = type_vnf_matrix

    def get_sfc_request(self):
        return self.sfc_request

    def set_sfc_request(self, sfc_request):
        self.sfc_request = sfc_request

    def get_vnf_order(self):
        return self.vnf_order

    def set_vnf_order(self, order):
        self.vnf_order = order

    # state状态初始化
    def init_state(self, remaining_cpu, remaining_bandwidth, distance):
        self.nodes_cpu = remaining_cpu
        # # 节点的出入带宽
        # self.nodes_bandwidth = self.calculate_bandwidth(remaining_bandwidth)
        # 节点之间的最短距离和带宽瓶颈
        self.nodes_distance, self.bandwidth_bottleneck = self.floyd_warshall_with_bandwidth(distance, remaining_bandwidth)
        self.nodes_bandwidth = self.calculate_bandwidth(remaining_bandwidth)

    def next_vnf(self, actions: Actions):
        self.vnf_order += 1
        actions.set_order(self.vnf_order)

    def next_sfc(self, actions: Actions, vnfs, bandwidth, source, destination):
        self.sfc_request = vnfs
        self.bandwidth_request = bandwidth
        self.ingress = source
        self.egress = destination
        self.vnf_order = 0
        actions.order = 0

    def reshape_state(self):
        nodes_cpu_reshape = self.nodes_cpu.reshape(1,-1).reshape(-1)
        nodes_bandwidth_reshape = self.nodes_bandwidth.reshape(1,-1).reshape(-1)
        bandwidth_bottleneck_reshape = self.bandwidth_bottleneck.reshape(1,-1).reshape(-1)
        type_vnf_matrix_reshape = self.type_vnf_matrix.reshape(1,-1).reshape(-1)
        sfc_request_reshape = self.sfc_request.reshape(1,-1).reshape(-1)
        vnf_order_reshape = np.array([self.vnf_order])
        bandwidth_request_reshape = self.bandwidth_request.reshape(1,-1).reshape(-1)
        ingress_reshape = self.ingress.reshape(1,-1).reshape(-1)
        egress_reshape = self.egress.reshape(1,-1).reshape(-1)

        result = np.concatenate((nodes_cpu_reshape, nodes_bandwidth_reshape))
        result = np.concatenate((result, bandwidth_bottleneck_reshape))
        result = np.concatenate((result, type_vnf_matrix_reshape))
        result = np.concatenate((result, sfc_request_reshape))
        result = np.concatenate((result, vnf_order_reshape))
        result = np.concatenate((result, bandwidth_request_reshape))
        result = np.concatenate((result, ingress_reshape))
        result = np.concatenate((result, egress_reshape))

        return result

