from exdata import *
import numpy as np
import dsd
import networkx as nx
from dsd import *
from LNPP import *
import gzip
from datetime import datetime
import matplotlib.pyplot as plt
from greedypeeling import charikar_peeling
from spectral import *
from baseline1_detection import *
from baseline2_lp import *

file_path = './datasets/ca-AstroPh.txt.gz'

#load dataset and show the graph
#G = nx.read_edgelist('./datasets/Facebook/facebook/1912.edges', nodetype=int)
with gzip.open(file_path, 'rt') as f:
    file_content = [next(f) for _ in range(10)]  # Read first 10 lines


# Create an empty graph
G = nx.Graph()

# Read the data and add edges to the graph
with gzip.open(file_path, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue  # Skip comment lines
        node1, node2 = map(int, line.strip().split('\t'))
        G.add_edge(node1, node2)
print("node is ",G.nodes)
print("the original graph density is",len(G.edges) / len(G.nodes))
print("G的边",G.edges)


#baseline1 没有加差分隐私的贪婪剥离算法
# print("greedypelling method")
# start = datetime.now()
# dense_subgraph_1,density_1 = charikar_peeling(G)
# # Print the nodes and edges of the dense subgraph
# t1 = datetime.now()-start
# print('run time', datetime.now()-start, '\n')
# print("Nodes in dense subgraph:", dense_subgraph_1.nodes())
# print("Density of the dense subgraph:", density_1)

#baseline2 没有加差分隐私的原始谱算法
print("Spectral method")
start = datetime.now()
dense_subgraph_2,density_2 = general_sweep_algorithm_laplacian_final(G,10)
t2 = datetime.now()-start
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph_2)
print("Density of the dense subgraph:", density_2)

#baseline3 采用指数机制进行选择的差分隐私贪婪剥离方法 对应的论文为“Differentially Private Densest Subgraph Detection”
print("The Exponential Mechanism DP greedypelling method")
start = datetime.now()
epsilon = 4.0
delta = 1e-5
dense_subgraph_3 = SEQDENSEDP(G, epsilon, delta)
t3 = datetime.now()-start
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph_3.nodes)
print("Density of the dense subgraph:", len(list(dense_subgraph_3.edges))/len((dense_subgraph_3.nodes)))

#baseline4 采用几何机制以及稀疏向量技术进行隐私保护的差分隐私贪婪剥离方法 对应的论文为“Differentially Private Densest Subgraph”
print("Symmetric geometric DP greedypelling method")
C = 10000
start = datetime.now()
dense_subgraph_4, density_4 = DP_Densest_Subgraph_Linear(G, epsilon, delta, C)
t4 = datetime.now()-start
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph_4)
print("Density of the dense subgraph:", density_4)

#our algorithm 采用差分隐私保护特征值与特征向量的谱最密子图差分隐私算法
print("our algorithm Spectral DP method")
start = datetime.now()
dense_subgraph_5,density_5 = general_sweep_algorithm_laplacian_final_DP(G,10)
t5 = datetime.now()-start
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph_5)
print("Density of the dense subgraph:", density_5, '\n')


# print("greedypelling method")
# print("Nodes in dense subgraph:", dense_subgraph_1.nodes())
# print("Density of the dense subgraph:", density_1)
# print("Run time is:",t1, '\n')

print("Spectral method")
print("Nodes in dense subgraph:", dense_subgraph_2)
print("Density of the dense subgraph:", density_2)
print("Run time is:",t2, '\n')

print("The Exponential Mechanism DP greedypelling method")
print("Nodes in dense subgraph:", dense_subgraph_3.nodes)
print("Density of the dense subgraph:", len(list(dense_subgraph_3.edges))/len((dense_subgraph_3.nodes)))
print("Run time is:",t3, '\n')

print("Symmetric geometric DP greedypelling method")
print("Nodes in dense subgraph:", dense_subgraph_4)
print("Density of the dense subgraph:", density_4)
print("Run time is:",t4, '\n')

print("our algorithm Spectral DP method")
print("Nodes in dense subgraph:", dense_subgraph_5)
print("Density of the dense subgraph:", density_5)
print("Run time is:",t5, '\n')