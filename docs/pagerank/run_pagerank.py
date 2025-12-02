import networkx as nx


def load_directed_graph(edge_list_path):
    G = nx.read_edgelist(
        edge_list_path,
        create_using=nx.DiGraph(),
        nodetype=int,
        comments="#",
    )
    return G


def load_undirected_as_bidirected(edge_list_path):
    G_und = nx.read_edgelist(
        edge_list_path,
        create_using=nx.Graph(),
        nodetype=int,
        comments="#",
    )
    G = nx.DiGraph()
    for u, v in G_und.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G


def pagerank_custom(G, d=0.85, tol=1.0e-6, max_iter=100):
    if not G.is_directed():
        G = G.to_directed()

    nodes = list(G.nodes())
    N = len(nodes)

    if N == 0:
        return {}

    pr = {n: 1.0 / N for n in nodes}

    for _ in range(max_iter):
        new_pr = {}
        dangling_mass = sum(pr[n] for n in nodes if G.out_degree(n) == 0)

        for i in nodes:
            rank = (1.0 - d) / N
            rank += d * dangling_mass / N

            for j in G.predecessors(i):
                out_deg_j = G.out_degree(j)
                if out_deg_j > 0:
                    rank += d * pr[j] / out_deg_j

            new_pr[i] = rank

        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr

        if diff < tol:
            break

    return pr


def top_k(pr_dict, k=10):
    return sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:k]


def run_experiment(edge_list_path, directed=True, ds=(0.5, 0.85, 0.99), tol=1e-6, max_iter=100):
    print("Iniciando experimento de PageRank...")

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
        print()


if __name__ == "__main__":
    EDGE_LIST_PATH = "data/Cit-HepTh.txt"
    run_experiment(EDGE_LIST_PATH, directed=True)
