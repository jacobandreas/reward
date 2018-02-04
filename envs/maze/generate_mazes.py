#!/usr/bin/env python3

from maze_utils import sample_maze

from pathlib import Path
import numpy as np

np.random.seed(1533)

for i in range(2000):
    art, actions = sample_maze()
    art_str = '\n'.join(art)
    act_str = ' '.join(str(a) for a in actions)
    with open(Path('data/maze.%d.txt' % i), 'w') as maze_f:
        print(art_str, file=maze_f)
        print(act_str, file=maze_f)
