from collections import namedtuple
import numpy as np
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

SLICE_SIZE = 5

class Batch(object):
    def __init__(self, obs, mstate, act, rew, cuda=False):
        assert (act is None) == (rew is None)
        assert (act is None) or (obs.shape[0] == act.shape[0] == rew.shape[0])
        self.obs = obs
        self.mstate = mstate
        self.act = act
        self.rew = rew

    def cuda(self):
        return Batch(
            self.obs.cuda(),
            None if self.mstate is None else self.mstate.cuda(),
            None if self.act is None else self.act.cuda(),
            None if self.rew is None else self.rew.cuda())

    @classmethod
    def from_obs(cls, obs):
        return Batch(
            Variable(FloatTensor(obs.reshape((obs.shape[0], 1, -1)))),
            None, None, None)

    @classmethod
    def from_bufs(cls, bufs, batch_size, slice_size):
        obs = np.zeros((batch_size, slice_size) + bufs[0][0].state.shape, dtype=np.int32)
        assert len(bufs[0][0].model_state.shape) == 1
        mstate = np.zeros((1, batch_size, bufs[0][0].model_state.shape[0]))
        act = np.zeros((batch_size, slice_size), dtype=np.int32)
        rew = np.zeros((batch_size, slice_size))
        for i, samp in enumerate(np.random.randint(len(bufs), size=batch_size)):
            episode = bufs[samp]
            offset = np.random.randint(max(len(episode) - slice_size, 1))
            for j in range(offset, min(offset + slice_size, len(episode))):
                transition = episode[j]
                bj = j - offset
                obs[i, bj, ...] = transition.state
                act[i, bj] = transition.action
                rew[i, bj] = transition.forward_reward
            mstate[0, i, :] = episode[offset].model_state
            obs[i, ...] = transition.state
            act[i] = transition.action
            rew[i] = transition.forward_reward
        return Batch(
            Variable(FloatTensor(obs.reshape((batch_size, slice_size, -1)))),
            Variable(FloatTensor(mstate)),
            Variable(LongTensor(act.astype(np.int64))),
            Variable(FloatTensor(rew)))

class Stats(object):
    def __init__(self, max_rew, min_rew, sum_rew, count):
        self.max_rew = max_rew
        self.min_rew = min_rew
        self.sum_rew = sum_rew
        self.count = count
        if count == 0:
            self.avg_rew = 0
        else:
            self.avg_rew = self.sum_rew / count

    @classmethod
    def of(cls, buf):
        rew = sum(t.reward for t in buf)
        return Stats(rew, rew, rew, 1)

    @classmethod
    def empty(cls):
        return Stats(0, 0, 0, 0)

    def update(self, other):
        if self.count == 0:
            self.max_rew = other.max_rew
            self.min_rew = other.min_rew
        else:
            self.max_rew = max(self.max_rew, other.max_rew)
            self.min_rew = min(self.min_rew, other.min_rew)
        self.sum_rew += other.sum_rew
        self.count += other.count
        self.avg_rew = self.sum_rew / self.count

    def items(self):
        return [
            ('avg_rew', self.avg_rew),
            ('max_rew', self.max_rew),
            ('min_rew', self.min_rew),
            ('count', self.count)
        ]

Transition = namedtuple('Transition', 
    ['state', 'model_state', 'action', 'reward', 'forward_reward'])
