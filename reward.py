#!/usr/bin/env python3

from envs import MetaRlWrapperEnv
from envs.maze import MazeEnv
from misc import hlog, fakeprof
from model import RnnModel
import rl

import gflags
import logging
import numpy as np
from pathlib import Path
import sys
import torch

FLAGS = gflags.FLAGS
gflags.DEFINE_string('run', None, 'task [train]')

def seed():
    np.random.seed(9063)
    torch.manual_seed(7352)

def get_cache():
    cache = Path('_cache')
    if not cache.exists():
        cache.mkdir()
    return cache

def main():
    seed()
    cache = get_cache()
    template_env = MetaRlWrapperEnv(MazeEnv(0))
    if FLAGS.run is None:
        raise Exception("'run' flag must be specified")
    elif FLAGS.run == 'train':
        model = RnnModel(template_env.featurizer, template_env.n_actions)
        rl.train(
            model,
            lambda: MetaRlWrapperEnv(MazeEnv(np.random.randint(1000))),
            lambda: MetaRlWrapperEnv(MazeEnv(1000 + np.random.randint(100))),
            cache / ('base.maze.txt'))
    else:
        raise Exception('no such task: %s' % FLAGS.task)

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
