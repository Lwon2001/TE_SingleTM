import networkx as nx
filename = './topology/Cer_topo.txt'
f = open(filename, 'r')
lines = f.readlines()
f.close()
lineList = lines[0].strip().split()  # 第一行保存了结点与边的数量信息
node_num = int(lineList[0])
link_num = int(lineList[1])
G = nx.Graph()
for i in range(1, link_num + 1):
    lineList = lines[i].strip().split()  # 第i行数据
    G.add_edge(int(lineList[0]), int(lineList[1]), weight=int(lineList[2]), init_weight=int(lineList[2]), capacity=float(lineList[3]))
ecmp_paths_dict = dict()
sss = [p for p in nx.all_shortest_paths(G, source=1, target=3, weight='weight')]
print('1')