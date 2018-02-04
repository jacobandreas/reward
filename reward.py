#!/usr/bin/env python3

from envs.maze import MazeEnv
from model import Model
import rl

import logging
import numpy as np

def main():
    template_env = MazeEnv(0)
    model = Model(template_env.n_features)
    rl.train(model, lambda: MazeEnv(0))

if __name__ == '__main__':
    main()
