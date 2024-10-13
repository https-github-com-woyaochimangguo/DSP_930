# Let's first examine the content of the uploaded file to understand its structure.
import gzip
import networkx as nx
# Path to the uploaded file
file_path = './datasets/ca-AstroPh.txt.gz'

# Let's read the first few lines of the file to understand its structure
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

# Get basic information about the graph
# 获取图的节点数和边数
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# 打印图的基本信息
print(f"Graph has {num_nodes} nodes and {num_edges} edges.")

