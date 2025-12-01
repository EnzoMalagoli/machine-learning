import networkx as nx


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
