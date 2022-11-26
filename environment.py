import networkx as nx
import numpy as np
import torch


class Env:
    def __init__(self, topo_name):
        self.G = nx.Graph()
        self.topo_name = topo_name
        self.node_num = 0
        self.link_num = 0
        self.TM_List = []
        self.TM_ptr = 0
        self.nodes = []
        self.node_degree = []
        self.link_capacity = []
        self.links = []
        self.init_MLU = 100
        self.action_low_space = 0.01
        self.action_high_space = 2
        self.demand_pair = []  # 保存流量需求的[src, dst]的结点对
        self.get_topo(topo_name)
        self.state = torch.zeros([self.link_num])  # 状态表示为链路利用率
        self.link_util = torch.zeros([self.link_num])
        self.action_dim = self.link_num
        self.state_dim = self.link_num
        self.init_env()

    def init_env(self):
        """利用初始TM和初始链路权重计算初始LU，获得环境初始状态，然后根据LU计算MLU，获得环境的初始MLU，后续用到该初始MLU来计算reward"""
        self.state = self.get_state(0)
        self.init_MLU = torch.max(self.state)

    def get_state(self, TM_ptr):
        """将TM_ptr对应的TM数据应用于当前的网络，采用OSPF路由策略计算LU，返回该LU作为状态"""
        TM = self.TM_List[TM_ptr]
        self.link_util = torch.zeros([self.link_num])
        # 遍历所有TM，每个TM按照最短路径路由
        demand_pair_ptr = 0
        for src, dst in self.demand_pair:
            # 为最短路径上的每条链接添加该流量值（ECMP）
            shortest_path_dict = [p for p in nx.all_shortest_paths(self.G, source=src, target=dst, weight='weight')]
            path_num = len(shortest_path_dict)
            for path in shortest_path_dict:
                ptr = 1  # 指针，用于遍历最短路径中的每条边
                traffic_demand = TM[demand_pair_ptr]  # 获取该TM下的src到dst的流量值
                while ptr < len(path):
                    # shortest_path[ptr-1] 与 shortest_path[ptr]分别为一条边的两个端节点
                    try:
                        index = self.links.index(
                            (path[ptr - 1], path[ptr]))  # index为对应的边在links中的下标
                    except ValueError:
                        index = self.links.index((path[ptr], path[ptr - 1]))
                    self.link_util[index] += traffic_demand / path_num  # 累计链路上的流量值
                    ptr += 1
        return torch.div(self.link_util, self.link_capacity)

    def get_topo(self, topo_name):
        filename = './topology/' + topo_name + '_topo.txt'
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        lineList = lines[0].strip().split()  # 第一行保存了结点与边的数量信息
        self.node_num = int(lineList[0])
        self.link_num = int(lineList[1])
        for i in range(1, self.link_num + 1):
            lineList = lines[i].strip().split()  # 第i行数据
            self.G.add_edge(int(lineList[0]), int(lineList[1]), weight=int(lineList[2]), init_weight=int(lineList[2]), capacity=float(lineList[3]))
        self.nodes = [node for node in self.G.nodes._nodes.keys()]
        self.node_degree = [len(self.G.degree._nodes[i]) for i in self.nodes]

        # 构建链路信息及链路容量信息
        for l in self.G.edges.items():
            self.links.append(l[0])
            self.link_capacity.append(l[1]['capacity'])
        self.link_capacity = torch.FloatTensor(self.link_capacity)

        # 读入所有TM需求
        filename = './Traffic_Matrix/fitting/' + self.topo_name + '_TMset_exp.txt'
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            lineList = line.strip().split(',')
            lineList = [float(i) for i in lineList]
            self.TM_List.append(lineList)

        # 构建demand_pair
        for src in self.nodes:
            for dst in self.nodes:
                if src == dst:
                    continue
                self.demand_pair.append((src, dst))

    def reset(self):
        """重置环境，并返回环境的初始状态"""
        self.TM_ptr = 0
        self.link_util = torch.zeros([self.link_num])
        for link in self.links:
            node1 = link[0]
            node2 = link[1]
            self.G.adj._atlas[node1][node2]['weight'] = self.G.adj._atlas[node1][node2]['init_weight']
        self.state = self.get_state(0)
        return self.state

    def set_weight(self, action):
        """
        根据agent的action设置网络的链路权重
        :param action:
        :return:
        """
        action = np.clip(action + 1.01, 0.1, 3)
        for i in range(0, self.link_num):
            node1 = self.links[i][0]
            node2 = self.links[i][1]
            self.G.adj._atlas[node1][node2]['weight'] = action[i]  # 为结点及其邻居之间的边赋新的权重

    def compute_reward(self, LU):
        """
        根据上一状态的TM，采取action的权重设置后，进行ospf路由，得到最大链路利用率，根据该链路利用率来给与相应奖励
        注意，在执行该函数前已经设置action对应的链路权重
        """
        MLU = torch.max(LU)
        alpha = self.init_MLU / MLU
        if alpha < 1:
            reward = -np.exp(2 * (1 / alpha - 1))
        elif alpha > 1:
            reward = np.exp(2 * (alpha - 1))
        else:
            reward = 0
        return reward

    def next_state(self, action):
        """
        :param action: agent做出的动作决策，具体表现为链路的权重设置
        :return: next_state,reward,done
        当网络接受到新的权重设置后，根据网络的最大链路利用率计算出reward
        然后根据新的权重设置生成新的TM，即为网络的下一个状态
        done始终为0
        """
        self.set_weight(action)
        self.TM_ptr += 1
        self.state = self.get_state(self.TM_ptr)
        reward = self.compute_reward(self.state)
        return [self.state, reward, 0]


if __name__ == '__main__':
    Env('Cer')
