import curses
import numpy as np
from pathlib import Path
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites
from pycolab import things

def _pos(position):
    return np.asarray([position.row, position.col])

class MazeEnv(object):
    def __init__(self, maze_id):
        self._game = None
        maze_path = Path(__file__).resolve().parent
        with open(Path(maze_path / ('data/maze.%d.txt' % maze_id))) as maze_f:
            data_lines = maze_f.readlines()
        art = data_lines[:-4]

        self._art = art
        self.n_actions = 4
        self.n_features = (
            (len(self._art) * len(self._art[0]) * 2)
            + self.n_actions
            + 1)

    def reset(self):
        game = ascii_art.ascii_art_to_game(
            self._art, what_lies_beneath=' ', sprites={
                'P': PlayerSprite,
                'G': GoalSprite,
                'D': DeathSprite,
            })
            
        self._game = game
        return self._process_step(self._game.its_showtime(), 0)

        #ui = human_ui.CursesUi(
        #    keys_to_actions={
        #        curses.KEY_UP: 0, curses.KEY_DOWN: 1, curses.KEY_LEFT: 2,
        #        curses.KEY_RIGHT: 3},
        #    delay=100)

        #ui.play(game)

    def step(self, action):
        return self._process_step(self._game.play(action), action)

    LAYERS = ['P', '#'] #+ [str(i) for i in range(10)]

    def _process_step(self, step, action):
        obs, rew, discount = step
        obs_data = np.zeros((len(self.LAYERS),) + obs.board.shape)
        for i, layer in enumerate(self.LAYERS):
            obs_data[i, ...] = obs.layers[layer]
        if rew is None:
            rew = 0
        act_data = np.zeros(self.n_actions)
        act_data[action] = 1
        rew_data = [rew]
        term = self._game._game_over
        features = np.concatenate((obs_data.ravel(), act_data, rew_data))
        return features, rew, term

    def done(self):
        return self._game._game_over

class GoalSprite(things.Sprite):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass

class DeathSprite(things.Sprite):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass

class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.init_goal_dist = None

    def update(self, actions, board, layers, backdrop, things, the_plot):
        goal_dist = sum(np.abs(_pos(self.position) - _pos(things['G'].position)))
        if self.init_goal_dist is None:
            self.init_goal_dist = goal_dist

	# apply motion commands.
        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._south(board, the_plot)
        elif actions == 2:
            self._west(board, the_plot)
        elif actions == 3:
            self._east(board, the_plot)

        post_goal_dist = sum(np.abs(_pos(self.position) - _pos(things['G'].position)))

        reward = (goal_dist - post_goal_dist) / self.init_goal_dist
        the_plot.add_reward(reward)
        if post_goal_dist == 0:
            the_plot.terminate_episode()

        #if self.position == things['G'].position:
        #    the_plot.add_reward(1)
        #    the_plot.terminate_episode()

        if self.position == things['D'].position:
            the_plot.add_reward(-1)
            the_plot.terminate_episode()
