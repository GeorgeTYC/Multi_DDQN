import numpy as np
from utils.Topology import Topology

class Utils:
    def __init__(self):
        return

    # 获取最近的卫星节点位置
    def get_nearest_node(self, topology: Topology, loc: np.ndarray):
        nodes_loc = topology.nodes_loc
        nodes_num = nodes_loc.shape[0]
        nearest_node = np.zeros((nodes_num, 1), dtype=bool)

        # 计算球面距离
        lat1, lon1 = nodes_loc[:, 0], nodes_loc[:, 1]
        lat2, lon2 = loc[0], loc[1]
        dlon = dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # 地球半径，单位为千米
        distances = R * c

        nearest_node_index = np.argmin(distances)
        nearest_node[nearest_node_index] = 1
        return nearest_node
