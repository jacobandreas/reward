#!/usr/bin/env python3

from maze import sample_maze

from pathlib import Path
import numpy as np

np.random.seed(1533)

for i in range(2000):
    print(i)
    art, opt_actions, hint, bad_actions, correction = sample_maze()
    art_str = '\n'.join(art)
    opt_act_str = ' '.join(str(a) for a in opt_actions)
    bad_act_str = ' '.join(str(a) for a in bad_actions)
    with open(Path('data/maze.%d.txt' % i), 'w') as maze_f:
        print(art_str, file=maze_f)
        print('opt', opt_act_str, file=maze_f)
        print('hint', hint, file=maze_f)
        print('bad', bad_act_str, file=maze_f)
        print('corr', correction, file=maze_f)
