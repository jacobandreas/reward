import numpy as np

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

PLAYER = 'P'
GOAL = 'G'
WALL = '#'
MARKERS = [str(i) for i in range(10)]

def shortest_path(board, init, goal):
    def neighbors(node):
        r, c = node
        out = []
        if r > 0 and not board[r-1, c]:
            out.append((NORTH, (r-1, c)))
        if r < board.shape[0] - 1 and not board[r+1, c]:
            out.append((SOUTH, (r+1, c)))
        if c > 0 and not board[r, c-1]:
            out.append((WEST, (r, c-1)))
        if c < board.shape[1] - 1 and not board[r, c+1]:
            out.append((EAST, (r, c+1)))
        return out

    stack = [((None, init),)]
    visited = set()
    while len(stack) > 0:
        history = stack.pop(0)
        action, node = history[-1]
        if node == goal:
            return history
        if node in visited:
            continue
        visited.add(node)
        for neighbor in neighbors(node):
            stack.append(history + (neighbor,))
    assert False

def random_maze(size):
    # crappy random spanning tree
    vert_edges = [((i,j), (i+1,j)) for i in range(size-1) for j in range(size)]
    horiz_edges = [((i,j), (i,j+1)) for i in range(size) for j in range(size-1)]
    edges = vert_edges + horiz_edges
    np.random.shuffle(edges)
    tree = []
    nodes = set(sum(edges, ()))
    groups = {n: {n} for n in nodes}
    while len(tree) < len(nodes) - 1:
        fr, to = edge = edges.pop()
        if fr in groups[to]:
            continue
        tree.append(edge)
        groups[fr] |= groups[to]
        for to_ in groups[to]:
            groups[to_] = groups[fr]

        groups[to] = groups[fr]
    assert groups[to] == nodes
    assert set(sum(tree, ())) == nodes
    return tree

def sample_maze():
    SIZE = 4
    true_size = SIZE * 2 + 1

    tree = random_maze(SIZE)
    board = np.zeros((true_size, true_size))

    # create walls
    board[0::2, :] = 1
    board[:, 0::2] = 1

    # knock out walls
    sp_tree = random_maze(SIZE)
    for fr, to in sp_tree:
        if fr[0] == to[0]:
            r = fr[0] * 2 + 1
            c = fr[1] * 2 + 2
        else:
            assert fr[1] == to[1]
            r = fr[0] * 2 + 2
            c = fr[1] * 2 + 1
        assert board[r, c] == 1
        board[r, c] = 0

    # place path and solve
    init = np.random.randint(SIZE, size=2)
    goal = None
    distractor = None

    while goal is None:
        goal = np.random.randint(SIZE, size=2)
        distractor = np.random.randint(SIZE, size=2)
        if (goal == init).all() or (distractor == init).all():
            goal = distractor = None
            continue
        raw_init = init * 2 + 1
        raw_goal = goal * 2 + 1
        raw_dist = distractor * 2 + 1
        demo = shortest_path(board, tuple(raw_init), tuple(raw_goal))

        dist_demo1 = shortest_path(board, tuple(raw_init), tuple(raw_dist))
        dist_demo2 = shortest_path(board, tuple(raw_dist), tuple(raw_goal))
        dist_cost1 = [np.abs(raw_goal - s).sum() for a, s in dist_demo1]
        dist_cost2 = [np.abs(raw_goal - s).sum() for a, s in dist_demo2]
        if min(dist_cost1) < 2 or dist_cost2[0] > 4 or (dist_cost2 - dist_cost2[0]).max() <= 0:
            goal = distractor = None
            continue

    # render
    art = [[' ' for _ in range(true_size)] for _ in range(true_size)]
    for i in range(true_size):
        for j in range(true_size):
            if board[i, j] == 1:
                art[i][j] = '#'
    art[raw_init[0]][raw_init[1]] = 'P'
    art[raw_goal[0]][raw_goal[1]] = 'G'

    counter = 0
    labels = []
    while counter < 10:
        pos = 2 * np.random.randint(SIZE, size=2) + 1
        r, c = pos
        if art[r][c] != ' ':
            continue
        art[r][c] = str(counter)
        labels.append((counter, pos))
        counter += 1

    nearest_goal, _ = min(labels, key=lambda x: np.abs(x[1] - raw_goal).sum())
    hint = 'near %d' % nearest_goal

    correction = None
    for i in range(min(len(demo), len(dist_demo1))):
        if demo[i] == dist_demo1[i]:
            continue
        good_a, _ = demo[i]
        _, common_s = demo[i-1]
        nearest_common, _ = min(labels, key=lambda x: np.abs(x[1] - common_s).sum())
        translate_a = {
            0: 'n',
            1: 's',
            2: 'w',
            3: 'e'
        }[good_a]

        correction = '%s near %d' % (translate_a, nearest_common)
        break
    if correction is None:
        correction = 'keep going'

    opt_actions = [a for a, s in demo]
    bad_actions = [a for a, s in dist_demo1]
    return (
        [''.join(r) for r in art],
        opt_actions[1:],
        hint,
        bad_actions[1:],
        correction)
