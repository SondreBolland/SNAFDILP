def all_subset_from(iterable):
    all_subsets = []
    for i in range(len(iterable)):
        subsets = list(combinations(iterable, i+1, p=2))
        all_subsets += subsets
    return all_subsets


def combinations(iterable, r, p=None):  # !!
    pool = iterable
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield list(pool[i] for i in indices)
    count = 1  # !!
    while p is None or count < p:  # !!
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield list(pool[i] for i in indices)
        count += 1  # !!


#subset = list(combinations(list([1,2,3]), 2, p=2))
print(list(all_subset_from([1,2,3,4])))