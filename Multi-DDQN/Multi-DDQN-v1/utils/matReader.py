import numpy as np
import scipy.io
import numpy as np
import math


# def get_topology_info(self, time):
#     distanceMap = scipy.io.loadmat('E:\projects\Multi-DDQN\Single-DDQN\data\distanceMap.mat')['distanceMap']
#     satellitesInfo = scipy.io.loadmat('E:\projects\Multi-DDQN\Single-DDQN\data\satellitesInfo.mat')['satellitesInfo']
#     print(distanceMap)

class matReader:
    def __init__(self, time):
        self.distanceMap = scipy.io.loadmat('E:\projects\Multi-DDQN\Multi-DDQN-v1\data\distanceMap.mat')['distanceMap']
        self.satellitesInfo = scipy.io.loadmat('E:\projects\Multi-DDQN\Multi-DDQN-v1\data\satellitesInfo.mat')[
            'satellitesInfo']
        self.time = time

    def get_topology_info(self):
        #links: 链路通断信息
        #links_distance: 链路距离
        n = self.distanceMap.shape[0]
        # 初始化连接信息，表示链路的连接情况，0-1
        links = np.zeros((n, n), dtype=np.int32)
        # 初始化链路长度信息
        links_distance = np.full((n, n), np.inf, dtype=np.float64)
        nodesIds = np.zeros((n, 1), dtype=np.int32)
        nodes_loc = np.zeros((n, 2), dtype=np.float64)

        #         print(self.satellitesInfo[0][1][0][1][1][0])
        #         print(self.satellitesInfo[0][2])

        for i in range(n):
            nodesIds[i, 0] = i
            node_i = np.array(
                (self.satellitesInfo[0][i][0][1][1][self.time], self.satellitesInfo[0][i][0][2][1][self.time]))
            nodes_loc[i, 0] = node_i[0]
            nodes_loc[i, 1] = node_i[1]
            for j in range(n):
                if self.distanceMap[i, j].size != 0:
                    node_j = np.array(
                        (self.satellitesInfo[0][j][0][1][1][self.time], self.satellitesInfo[0][j][0][2][1][self.time]))
                    distance = self.get_distance(node_i, node_j)
                    links[i, j], links[j, i] = 1, 1
                    links_distance[i, j], links_distance[j, i] = distance, distance

        return nodesIds, links, links_distance, nodes_loc

    # 在已知两点经纬度的前提下计算距离
    def get_distance(self, x, y):
        # x: A点的经纬度【经度 纬度】
        # y: B点的经纬度 【经度 纬度】
        # R：距离
        R = 7378.137
        Deltas = math.acos(math.cos(x[1]) * math.cos(y[1]) * math.cos(x[0] - y[0]) + math.sin(x[1]) * math.sin(y[1]))
        d = R * Deltas
        return d

# matReader = matReader(0)
# nodesIds, links, links_distance, nodes_loc = matReader.get_topology_info()

# print(nodes_loc)
# print(nodesIds)
# print(links)