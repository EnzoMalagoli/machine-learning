import networkx as nx


def load_directed_graph(edge_list_path):
    G = nx.read_edgelist(
        edge_list_path,
        create_using=nx.DiGraph(),
        nodetype=int,
        comments="#"
    )
    return G


def load_undirected_as_bidirected(edge_list_path):
    G_und = nx.read_edgelist(
        edge_list_path,
        create_using=nx.Graph(),
        nodetype=int,
        comments="#"
    )
    G = nx.DiGraph()
    for u, v in G_und.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G
