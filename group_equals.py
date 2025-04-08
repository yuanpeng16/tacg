def group_equals(pairs):
    parent = {}
    rank = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        if x not in parent:
            parent[x] = x
            rank[x] = 1
        if y not in parent:
            parent[y] = y
            rank[y] = 1
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
                if rank[root_x] == rank[root_y]:
                    rank[root_y] += 1

    for a, b in pairs:
        union(a, b)

    elements = set()
    for a, b in pairs:
        elements.add(a)
        elements.add(b)

    groups = {}
    for elem in elements:
        root = find(elem)
        if root not in groups:
            groups[root] = []
        groups[root].append(elem)

    return list(groups.values())


def group_equivalent_pairs(pairs):
    result = group_equals(pairs)
    return [set(x) for x in result]


if __name__ == '__main__':
    my_pairs = [(1, 1), (2, 3), (3, 2), (4, 4)]
    print(group_equivalent_pairs(my_pairs))  # output：[[1], [2, 3], [4]]
