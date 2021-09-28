def neg_cycle(G):
    visited = [False] * len(G)
    for v in G:
        rec_stack = [v]
        visited[v] = True
        if find_neg_cycle(G, v, visited, rec_stack):
            return True

def find_neg_cycle(G, v, visited, rec_stack):
    if rec_stack.contains(v):
        return has_negative(rec_stack)
    if visited[v]:
        return False

    visited[v] = True
    rec_stack.append(v)

    children = G.get_children(v)
    for u in children:
        if find_neg_cycle(G, u, visited, rec_stack):
            return True

    rec_stack.remove(v)
    return False

def has_negative(rec_stack):
    for v in rec_stack:
        if v.is_negative():
            return True
    return False