#!/usr/bin/env python3

from envs.meta import MetaFeatureEnv
from envs.maze import MazeEnv
from misc import hlog, fakeprof
from model import RnnModel, SimpleModel
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
    meta_featurizer = lambda env: MetaFeatureEnv(env)
    cache = get_cache()
    template_env = MazeEnv(0)
    template_meta_env = meta_featurizer(template_env)
    if FLAGS.run is None:
        raise Exception("'run' flag must be specified")
    elif FLAGS.run == 'train':
        model1 = RnnModel(template_meta_env.featurizer, template_env.n_actions)
        model2 = SimpleModel(template_env.featurizer, template_env.n_actions)
        rl.train_meta(
            model1,
            model2,
            lambda: MazeEnv(np.random.randint(1000)),
            lambda: MazeEnv(1000 + np.random.randint(100)),
            meta_featurizer,
            cache / ('base.maze.txt'))
    else:
        raise Exception('no such task: %s' % FLAGS.task)

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    main()
