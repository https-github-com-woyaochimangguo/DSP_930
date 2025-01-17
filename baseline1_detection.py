import networkx as nx
import numpy as np


# 引入上面的SEQDENSEDP和相关函数
def SEQDENSEDP(graph, epsilon, delta):
    n = len(graph)  # number of nodes
    epsilon_prime = epsilon / (4 * np.log(np.e / delta))
    S = list(graph.nodes)
    candidate_subgraphs = []  # 用于保存候选子集（子图对象）

    # Iteratively remove nodes with probability proportional to their degree using Exponential Mechanism
    for t in range(n):
        degrees = {v: len(list(graph.neighbors(v))) for v in S}
        # Use the exponential mechanism to choose a node with probability proportional to exp(-epsilon_prime * degree)
        scores = np.array([-degrees[v] for v in S])  # Here we use negative degree as score (lower is better)
        probabilities = np.exp(epsilon_prime * scores)
        probabilities /= probabilities.sum()  # Normalize probabilities
        removed_node = np.random.choice(S, p=probabilities)
        S.remove(removed_node)

        # Add the current subgraph (as node list) to the candidate subgraph list
        candidate_subgraphs.append(S.copy())  # 存储节点列表

    # Calculate the density of each candidate subgraph
    subgraph_densities = [(i, calculate_density(graph.subgraph(subgraph))) for i, subgraph in
                          enumerate(candidate_subgraphs)]

    # Use the exponential mechanism to select the best subgraph based on their densities
    densities = np.array([density for _, density in subgraph_densities])
    probabilities = np.exp(epsilon * densities / 2)
    probabilities /= probabilities.sum()  # Normalize probabilities

    # Select the index of a subgraph from the candidate subgraphs with probability proportional to e^(epsilon * density / 2)
    selected_index = np.random.choice([i for i, _ in subgraph_densities], p=probabilities)

    # Return the selected subgraph (as a graph object)
    return graph.subgraph(candidate_subgraphs[selected_index])


def calculate_density(subgraph):
    # density = (#edges in subgraph) / (#nodes in subgraph)
    num_nodes = len(subgraph.nodes)
    if num_nodes == 0:
        return 0
    return len(subgraph.edges) / num_nodes


# 创建测试图
def create_test_graph():
    G = nx.Graph()
    # 添加节点和边
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (2, 4)]
    G.add_edges_from(edges)
    return G


# 测试算法
def test_SEQDENSEDP():
    # 创建一个简单的无向图
    G = create_test_graph()

    # 打印原始图的边和节点
    print("Original Graph:")
    print("Nodes:", G.nodes)
    print("Edges:", G.edges)

    # 设置差分隐私参数
    epsilon = 1.0
    delta = 1e-5

    # 运行算法
    best_subgraph = SEQDENSEDP(G, epsilon, delta)

    # 打印结果
    print("\nBest Subgraph Selected:")
    print("Nodes:", best_subgraph.nodes)
    print("Edges:", best_subgraph.edges)
    print("选出的最密子图的密度为",len(list(best_subgraph.edges))/len((best_subgraph.nodes)))


# 运行测试
