import networkx as nx
from pagerank_custom import pagerank_custom
from load_graph import load_directed_graph, load_undirected_as_bidirected


def top_k(pr_dict, k=10):
    return sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:k]


def run_experiment(edge_list_path, directed=True, ds=(0.5, 0.85, 0.99), tol=1e-6, max_iter=100):
    if directed:
        G = load_directed_graph(edge_list_path)
    else:
        G = load_undirected_as_bidirected(edge_list_path)

    print(f"Grafo carregado: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.\n")

    for d in ds:
        print(f"=== PageRank com damping factor d = {d} ===")

        pr_custom = pagerank_custom(G, d=d, tol=tol, max_iter=max_iter)
        pr_nx = nx.pagerank(G, alpha=d, tol=tol)

        all_nodes = pr_custom.keys()
        max_diff = max(abs(pr_custom[n] - pr_nx[n]) for n in all_nodes)
        print(f"Máxima diferença entre custom e networkx: {max_diff:.2e}")

        top10 = top_k(pr_custom, k=10)
        print("Top 10 nós (nó, PageRank):")
        for node, score in top10:
            print(f"{node}\t{score:.6f}")
        print("\n")


if __name__ == "__main__":
    EDGE_LIST_PATH = "data/Cit-HepTh.txt"
    run_experiment(EDGE_LIST_PATH, directed=True)
