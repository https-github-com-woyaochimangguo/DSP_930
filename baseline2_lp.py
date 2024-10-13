import numpy as np
import networkx as nx
from collections import defaultdict


# 定义几何分布噪声生成函数
def geometric_noise(epsilon):
    # 使用几何分布生成噪声的绝对值
    noise = np.random.geometric(p=1 - np.exp(-epsilon))
    # 返回对称的噪声
    return  noise

# 定义桶排序结构
class Bucket:
    def __init__(self):
        self.buckets = defaultdict(list)  # 存储每个桶中的节点
        self.node_positions = {}  # 记录每个节点在桶中的位置

    def add_node(self, node, bucket_idx):
        self.buckets[bucket_idx].append(node)
        self.node_positions[node] = bucket_idx

    def remove_node(self, node):
        bucket_idx = self.node_positions[node]
        self.buckets[bucket_idx].remove(node)
        del self.node_positions[node]

    def get_smallest_non_empty_bucket(self, start_idx):
        for idx in range(start_idx, -1, -1):
            if self.buckets[idx]:
                return idx
        return None


# 差分隐私的最密子图检测算法
def DP_Densest_Subgraph_Linear(graph, epsilon, sigma, C):
    n = len(graph)  # 节点数量
    T = (C / epsilon) * np.log(n) * np.log(1 / sigma)  # 计算 T 值
    epsilon0 = epsilon1 = epsilon2 = epsilon_prime = epsilon / 4

    # 初始化桶
    bucket = Bucket()
    S = list(graph.nodes)
    d_max = 0
    S_star = S.copy()

    # 初始化每个节点的计数器
    D = {v: graph.degree(v) for v in S}  # 度数
    Cnt = {v: 0 for v in S}  # Cnt(u)
    PSum = {v: 0 for v in S}  # PSum(u)
    E = {v: geometric_noise(epsilon2) for v in S}  # E(u)

    # 初始化每个节点放入合适的桶中
    for v in S:
        bucket.add_node(v, D[v])

    while S:
        # 获取上一步移除节点的桶索引
        idx_prime = bucket.get_smallest_non_empty_bucket(len(S))
        if idx_prime is None:
            break

        # 获取最小非空桶中的节点并移除
        idx = bucket.get_smallest_non_empty_bucket(idx_prime)
        v = bucket.buckets[idx][0]  # 获取桶中的第一个节点
        bucket.remove_node(v)
        S.remove(v)

        # 更新最大密度和 S_star
        if d_max < D[v] - PSum[v]:
            d_max = D[v] - PSum[v]
            S_star = S.copy()

        # 对邻居节点更新 Cnt(u)
        for u in graph.neighbors(v):
            if u in S:
                Cnt[u] += 1

        # 对每个节点 u 更新 PSum(u) 和桶中的位置
        for u in S:
            if Cnt[u] + E[u] + geometric_noise(epsilon2) > T:
                PSum[u] += Cnt[u]
                new_d = D[u] - PSum[u]
                old_bucket_idx = bucket.node_positions[u]
                bucket.remove_node(u)
                bucket.add_node(u, new_d)
                Cnt[u] = 0
                E[u] = geometric_noise(epsilon2)

    # 最终返回的子集 S_star 和密度 d_star
    geom_noise = geometric_noise(epsilon_prime)
    d_star = min((len(graph.edges(S_star)) + geom_noise) / len(S_star), len(S_star))

    return S_star, d_star


# 测试函数
def test_DP_Densest_Subgraph_Linear(G):
    epsilon = 1.0
    sigma = 0.1
    C = 10  # 常数

    S_star, d_star = DP_Densest_Subgraph_Linear(G, epsilon, sigma, C)

    print("Selected Subgraph Nodes:", S_star)
    print("Density of the Subgraph:", d_star)


# 运行测试