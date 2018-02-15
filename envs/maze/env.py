from .maze import NORTH, SOUTH, WEST, EAST
from .maze import PLAYER, GOAL, WALL, MARKERS

import curses
import gflags
import numpy as np
from pathlib import Path
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites
from pycolab import things

import torch
from torch import nn

SCORE = 'X'
SIMPLE_FEATURES = [PLAYER, WALL]
LANDMARK_FEATURES = [PLAYER, WALL] + MARKERS

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean('show_landmarks', False, 'agent sees landmark features')

def _pos(position):
    return np.asarray([position.row, position.col])

def _dist(p1, p2):
    return np.abs(_pos(p1) - _pos(p2)).sum()

class ConvMazeFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurize_board = nn.Sequential(
            nn.Conv2d(2, 16, 3),
            nn.Tanh()
            )
        self.n_output = 789

    def forward(self, obs):
        board, others = obs
        n_batch, n_time, *rest = board.shape
        board = board.view((n_batch * n_time,) + tuple(rest))
        board_feats = self.featurize_board(board)
        board_feats = board_feats.view((n_batch, n_time, -1))
        return torch.cat((board_feats, others), dim=2)

    def extend(self, new_feature_shape):
        assert False

class MlpFeaturizer(nn.Module):
    N_HIDDEN = 64
    def __init__(self, feature_shape):
        super().__init__()
        self.n_output = self.N_HIDDEN
        self._n_features = int(sum(np.prod(s) for s in feature_shape))
        self._featurize = nn.Sequential(
            nn.Linear(self._n_features, self.N_HIDDEN),
            nn.Tanh()
            )

    def forward(self, obs):
        n_batch, n_time = obs[0].shape[:2]
        parts = [o.view(n_batch, n_time, -1) for o in obs]
        feats = torch.cat(parts, dim=2)
        return self._featurize(feats)

    def extend(self, new_feature_shape):
        return MlpFeaturizer(new_feature_shape)

#art1 = [
#    "#########",
#    "#dG P D #",
#    "#########"]
#art2 = [
#    "#########",
#    "# d P GD#",
#    "#########"]

class MazeEnv(object):
    def __init__(self, maze_id, val=False):
        self._game = None
        self._val = val

        maze_path = Path(__file__).resolve().parent
        with open(Path(maze_path / ('data/maze.%d.txt' % maze_id))) as maze_f:
            data_lines = maze_f.readlines()
        art = [l.strip() for l in data_lines[:-4]]
        self._art = art
        #self._art = art1 if maze_id % 2 == 0 else art2
        self._features = LANDMARK_FEATURES if FLAGS.show_landmarks else SIMPLE_FEATURES

        self.n_actions = 4
        #self.feature_shape = ((9, 9, len(self._features)),)
        self.feature_shape = ((len(self._features), len(self._art), len(self._art[0])),)

        self.featurizer = MlpFeaturizer(self.feature_shape)
        #self.featurizer = ConvMazeFeaturizer()

    def reset(self):
        game = ascii_art.ascii_art_to_game(
            self._art, 
            what_lies_beneath=' ',
            sprites={
                PLAYER: PlayerSprite,
                GOAL: GoalSprite,
                'D': DeathSprite,
                'd': DeathSprite
            },
            drapes={
                SCORE: ScoreDrape,
            },
            update_schedule=[PLAYER, GOAL, 'D', 'd', SCORE])

        #ui = human_ui.CursesUi(
        #    keys_to_actions={
        #        curses.KEY_UP: 0,
        #        curses.KEY_DOWN: 1,
        #        curses.KEY_LEFT: 2,
        #        curses.KEY_RIGHT: 3},
        #    delay=100)
        #ui.play(game)
            
        self._game = game
        return self._process_step(self._game.its_showtime())

    def step(self, action):
        return self._process_step(self._game.play(action))

    def _process_step(self, step):
        obs, rew, discount = step
        obs_data = np.zeros((len(self._features),) + obs.board.shape)
        for i, layer in enumerate(self._features):
            obs_data[i, ...] = obs.layers[layer]
        features = (obs_data,)
        rew = rew if rew is not None else 0
        term = self._game._game_over
        return features, rew, term

    def done(self):
        return self._game._game_over

class GoalSprite(things.Sprite):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass

class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=WALL)

    def update(self, actions, board, layers, backdrop, things, the_plot):
          # apply motion commands.
        if actions == NORTH:
            self._north(board, the_plot)
        elif actions == SOUTH:
            self._south(board, the_plot)
        elif actions == WEST:
            self._west(board, the_plot)
        elif actions == EAST:
            self._east(board, the_plot)

class ScoreDrape(things.Drape):
    def __init__(self, curtain, character):
        super().__init__(curtain, character)
        self._init_player_pos = None
        self._player_pos = None
        self._init_dist = None

    def update(self, actions, board, layers, backdrop, things, the_plot):
        new_player_pos = things[PLAYER].position
        goal_pos = things[GOAL].position
        if self._player_pos is None:
            self._player_pos = new_player_pos
            self._init_player_pos = new_player_pos
            self._init_dist = _dist(new_player_pos, goal_pos)
            return

        old_player_pos = self._player_pos
        old_dist = _dist(old_player_pos, goal_pos)
        new_dist = _dist(new_player_pos, goal_pos)
        delta = (old_dist - new_dist) / self._init_dist
        the_plot.add_reward(delta)
        if new_player_pos == goal_pos:
            the_plot.terminate_episode()

        self._player_pos = new_player_pos

class DeathSprite(things.Sprite):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        if things['P'].position == self.position:
            the_plot.add_reward(-1)
            the_plot.terminate_episode()
