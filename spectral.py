import warnings
from LNPP import *
import numpy as np
import networkx as nx
from scipy.linalg import eigh


def compute_density(G, S):
    """
    Compute the density of the subgraph induced by the set S.

    Parameters:
    G (networkx.Graph): The input graph.
    S (set): A subset of nodes in the graph.

    Returns:
    float: The density of the subgraph induced by S.
    """
    subgraph = G.subgraph(S)
    num_edges = subgraph.number_of_edges()
    num_nodes = subgraph.number_of_nodes()
    return num_edges / num_nodes if num_nodes > 0 else 0

# def general_sweep_algorithm(G):
#     """
#     General Sweep Algorithm to find the densest subgraph.
#
#     Parameters:
#     G (networkx.Graph): The input graph.
#
#     Returns:
#     tuple: A tuple containing the densest subgraph and its density.
#     """
#     adj_matrix = nx.adjacency_matrix(G).todense()
#     eigenvalues, eigenvectors = eigh(adj_matrix)
#     v1 = eigenvectors[:, -1]  # Main eigenvector
#
#     # Sort nodes in nonincreasing order of v1(i)
#     sorted_nodes = np.argsort(-v1)
#
#     best_S = set()
#     best_density = 0
#
#     for s in range(1, len(sorted_nodes) + 1):
#         S = set(sorted_nodes[:s])
#         current_density = compute_density(G, S)
#
#         if current_density > best_density:
#             best_S = S
#             best_density = current_density
#
#     return best_S, best_density

def general_sweep_algorithm(G):
    """
    General Sweep Algorithm to find the densest subgraph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the densest subgraph and its density.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        adj_matrix = nx.adjacency_matrix(G)  # Returns a scipy.sparse matrix
    # adj_matrix = nx.adjacency_matrix(G)  # Returns a scipy.sparse matrix
    adj_matrix_dense = adj_matrix.toarray()  # Convert to a dense matrix
    eigenvalues, eigenvectors = eigh(adj_matrix_dense)
    v1 = eigenvectors[:, -1]  # Main eigenvector
    print("the main eigenvectors is", v1)
    # Sort nodes in nonincreasing order of v1(i)
    sorted_nodes = np.argsort(-v1)
    print("the sorted nodes list is ", sorted_nodes)

    best_S = set()
    best_density = 0

    for s in range(1, len(sorted_nodes) + 1):
        S = set(sorted_nodes[:s])
        current_density = compute_density(G, S)

        if current_density > best_density:
            best_S = S
            best_density = current_density

    return best_S, best_density


def general_sweep_algorithm_laplacian(G):
    """
    General Sweep Algorithm to find the densest subgraph using Laplacian matrix.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the subset of nodes in the densest subgraph found and its density.
    """
    # Compute the Laplacian matrix
    laplacian_matrix = nx.laplacian_matrix(G).todense()
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    # Use the second smallest eigenvector (Fiedler vector)
    v1 = eigenvectors[:, 1]
    print("the main eigenvectors is",v1)

    # Sort nodes in nonincreasing (descending) order of v1(i)
    sorted_nodes = np.argsort(v1)
    print("the sorted nodes list is ",sorted_nodes)
    best_S = set()
    best_density = 0

    for s in range(1, len(sorted_nodes) + 1):
        S = set(sorted_nodes[:s])
        print("现在的子集S",S)
        current_density = compute_density(G, S)
        print("现在子集的密度：",current_density)

        # Update best subset if current density is greater
        if current_density > best_density:
            best_S = S
            best_density = current_density
        print("现在子集的密度：", current_density)

    return best_S, best_density  # Return both the subset and its density


#标准的不加噪声的扫描方法 BASELINE1
def general_sweep_algorithm_laplacian_final(G,k):
    """
    General Sweep Algorithm to find the densest subgraph using Laplacian matrix.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the subset of nodes in the densest subgraph found and its density.
    """
    # Compute the Laplacian matrix
    laplacian_matrix = nx.laplacian_matrix(G).todense()
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    print("未排序的特征值:")
    print(eigenvalues)
    print("\n未排序的特征向量:")
    print(eigenvectors)
    # Use the second smallest eigenvector (Fiedler vector)
    # sorted_indices = np.argsort(eigenvalues)
    # print("特征值大小的排序为：",sorted_indices)
    # print("排序后的特征值列表：",eigenvalues[sorted_indices])
    best_S = list()
    best_density = 0
    #for x in range(0,len(eigenvalues)):
    for x in range(0,k):
        v1 = eigenvectors[:,-x]
        #print("当前特征值的大小为：",eigenvalues[-x])
        #print("当前特征值对应的特征向量:",eigenvectors[:,-x])
        # Sort nodes in nonincreasing (descending) order of v1(i)
        sorted_nodes = np.argsort(v1)
        nodes_sorted = []
        G_nodes=list(G.nodes)
        #循环访问对应特征值最大的点
        for i in sorted_nodes[::-1]:
            nodes_sorted.append(G_nodes[i])

        current_S = list()  # Initialize the current set
        node = list()
        for node in nodes_sorted:
            current_S.append(node)  # Add the node to the current set
            # print("现在的子集S", current_S)
            # print("生成的子图的边",list(G.subgraph(current_S).edges))
            current_density = compute_density(G, current_S)  # Compute the density
            #print("current density is",current_density)
            # Update best subset if current density is greater
            if current_density > best_density:
                best_S = current_S.copy()  # Update best set
                best_density = current_density  # Update best density
        #print("现在子集的密度：", best_density)
        #print("现在子集的节点为：",best_S)

    return best_S, best_density  # Return both the subset and its density

#加噪声的扫描方法
def general_sweep_algorithm_laplacian_final_DP(G,k):
    """
    General Sweep Algorithm under DP to find the densest subgraph using Laplacian matrix.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the subset of nodes in the densest subgraph found and its density.
    """
    # Compute the Laplacian matrix
    #laplacian_matrix = nx.laplacian_matrix(G).todense()
    # Compute eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = eigh(laplacian_matrix)
    epsilon = 10000
    eigenvalues, eigenvectors = lnpp(G, epsilon, k)
    print("未排序的加噪特征值:")
    print(eigenvalues)
    print("\n未排序的加噪特征向量:")
    print(eigenvectors)
    # Use the second smallest eigenvector (Fiedler vector)
    #sorted_indices = np.argsort(eigenvalues)
    #print("加噪特征值大小的排序为：",sorted_indices)
    #print("排序后的加噪特征值列表：",eigenvalues[sorted_indices])
    best_S = list()
    best_density = 0
    for x in range(0,k):
        v1 = eigenvectors[:,-x]
        print("当前特征值对应的特征向量:",eigenvectors[:,-x])
        # Sort nodes in nonincreasing (descending) order of v1(i)
        values = np.argsort(v1)
        nodes = list(np.arange(0,len(G.nodes)))
        value_node_pairs = list(zip(values, nodes))
        sorted_pairs = sorted(value_node_pairs, key=lambda x: x[0])
        sorted_nodes = [node for value, node in sorted_pairs]
        nodes_sorted = []
        G_nodes=list(G.nodes)
        for i in sorted_nodes[::-1]:
            nodes_sorted.append(G_nodes[i])
        print("sorted_node lists is",nodes_sorted)


        current_S = list()  # Initialize the current set
        node = list()
        for node in nodes_sorted:
            current_S.append(node)  # Add the node to the current set
            # print("现在的子集S", current_S)
            # print("生成的子图的边",list(G.subgraph(current_S).edges))
            current_density = compute_density(G, current_S)  # Compute the density
            #print("current density is",current_density)
            # Update best subset if current density is greater
            if current_density > best_density:
                best_S = current_S.copy()  # Update best set
                best_density = current_density  # Update best density
        print("现在子集的密度：", best_density)

    return best_S, best_density  # Return both the subset and its density