from misc import hlog

from collections import namedtuple
import numpy as np
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

class Batch(object):
    def __init__(self, obs, mstate, act, rew, mask, lens, cuda=False):
        assert isinstance(obs, tuple)
        assert (act is None) == (rew is None)
        assert (act is None) or (obs[0].shape[0] == act.shape[0] == rew.shape[0])
        self.obs = obs
        self.mstate = mstate
        self.act = act
        self.rew = rew
        self.mask = mask
        self.lens = lens

    def cuda(self):
        return Batch(
            tuple(o.cuda() for o in self.obs),
            None if self.mstate is None else self.mstate.cuda(),
            None if self.act is None else self.act.cuda(),
            None if self.rew is None else self.rew.cuda(),
            None if self.mask is None else self.mask.cuda(),
            None if self.lens is None else self.lens.cuda())

    @classmethod
    def from_obs(cls, obs):
        assert isinstance(obs[0], tuple)
        out = []
        for i_part in range(len(obs[0])):
            part = np.asarray([o[i_part] for o in obs])
            var = Variable(FloatTensor(part.reshape((part.shape[0], 1) + part.shape[1:])))
            out.append(var)
        return Batch(tuple(out), None, None, None, None, None)

    @classmethod
    def from_bufs(cls, bufs, batch_size, slice_size):
        assert isinstance(bufs[0][0][0].state, tuple)
        n_bufs = len(bufs[0])
        assert all(len(b) == n_bufs for b in bufs)
        assert len(bufs[0][0][0].model_state.shape) == 1

        obs = [
            tuple(np.zeros((batch_size, slice_size) + o.shape) for o in b[0][0].state)
            for b in bufs]
        mstate = [
            np.zeros((batch_size, b[0][0].model_state.shape[0]))
            for b in bufs]
        act = [np.zeros((batch_size, slice_size), dtype=np.int32) for _ in bufs]
        rew = [np.zeros((batch_size, slice_size)) for _ in bufs]
        mask = [np.zeros((batch_size, slice_size)) for _ in bufs]
        lens = [np.zeros((batch_size,)) for _ in bufs]

        indices = range(n_bufs) if n_bufs == batch_size else np.random.randint(n_bufs, size=batch_size)

        for i, samp in enumerate(indices):
            for j in range(len(bufs)):
                episode = bufs[j][samp]
                for k in range(len(episode)):
                    transition = episode[k]
                    for i_part in range(len(transition.state)):
                        obs[j][i_part][i, k, ...] = transition.state[i_part]
                    act[j][i, k] = transition.action
                    rew[j][i, k] = transition.forward_reward
                    mask[j][i, k] = 1
                lens[j][i] = len(episode)
            ## episode = bufs[samp]
            ## offset = np.random.randint(max(len(episode) - slice_size, 1))
            ## assert offset == 0
            ## for j in range(offset, min(offset + slice_size, len(episode))):
            ##     transition = episode[j]
            ##     bj = j - offset
            ##     for i_part in range(len(transition.state)):
            ##         obs[i_part][i, bj, ...] = transition.state[i_part]
            ##     act[i, bj] = transition.action
            ##     rew[i, bj] = transition.forward_reward
            ##     mask[i, bj] = 1
            ##     # TODO LOCALITY!!!
            ##     #mstate[0, i, :] = episode[offset].model_state
        return [
            Batch(
                tuple(Variable(FloatTensor(o)) for o in obs[j]),
                Variable(FloatTensor(mstate[j])),
                Variable(LongTensor(act[j].astype(np.int64))),
                Variable(FloatTensor(rew[j])),
                Variable(FloatTensor(mask[j])),
                Variable(LongTensor(lens[j])))
            for j in range(len(bufs))]

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
