from misc import hlog

from collections import Counter
import numpy as np
from torch import FloatTensor, LongTensor
from torch.autograd import Variable

# TODO magic
N_BATCH = 5000

class EnvWrapper(object):
    def __init__(self, env_builder):
        self.underlying = env_builder()

    def reset(self):
        return self.underlying.reset()

    def step(self, action):
        if self.underlying.done():
            return (None, None, None)
        return self.underlying.step(action)

class Batch(object):
    def __init__(self, obs, act=None, rew=None):
        assert (act is None) == (rew is None)
        assert (act is None) or (obs.shape[0] == act.shape[0] == rew.shape[0])

        obs_arr = obs

        obs_arr = obs_arr.reshape((obs_arr.shape[0], -1))
        self.obs = Variable(FloatTensor(obs_arr))
        if act is None:
            self.act = None
            self.rew = None
        else:
            self.act = Variable(LongTensor(act.astype(np.int64)))
            self.rew = Variable(FloatTensor(rew))

    @classmethod
    def from_buf(cls, buf, size):
        obs = np.zeros((size,) + buf[0][0].shape, dtype=np.int32)
        act = np.zeros((size,), dtype=np.int32)
        rew = np.zeros((size,))
        for i, samp in enumerate(np.random.randint(len(buf), size=size)):
            s_obs, s_act, s_rew, s_mc_rew = buf[samp]
            obs[i, ...] = s_obs
            act[i] = s_act
            rew[i] = s_mc_rew
        return Batch(obs, act, rew)

def rollout(model, envs):
    steps = [e.reset() for e in envs]
    obs, rew, term = zip(*steps)
    done = list(term)
    bufs = [[] for _ in envs]
    # TODO magic
    for t in range(100):
        if all(done):
            break
        obs_data = np.asarray(obs)
        batch = Batch(obs_data)
        act = model.act(batch)
        steps = [e.step(a) for e, a in zip(envs, act)]
        obs_, rew, term = zip(*steps)
        for i in range(len(obs_)):
            if done[i]:
                continue
            done[i] = done[i] or term[i]
            bufs[i].append((obs[i], act[i], rew[i]))
        obs = [o_ if o_ is not None else o for o, o_ in zip(obs, obs_)]

    mc_bufs = []
    stats = Counter()
    for buf in bufs:
        mc_rew = 0
        mc_buf = []
        for obs, act, rew in buf[::-1]:
            # TODO magic
            mc_rew = mc_rew * 0.95 + rew
            mc_buf.append((obs, act, rew, mc_rew))
            stats['rew'] += rew
        mc_bufs.append(mc_buf)

    return mc_bufs, stats

def train(model, env_builder):
    # TODO magic
    envs = [EnvWrapper(env_builder) for _ in range(10)]
    for i_epoch in hlog.loop('epoch_%05d', range(1000)):
        buf = []
        stats = Counter()
        count = 0
        while len(buf) < N_BATCH:
            rbuf, rstats = rollout(model, envs)
            for b in rbuf:
                buf += b
            stats += rstats
            count += len(envs)
        batch = Batch.from_buf(buf, N_BATCH)
        loss = model.train(batch)
        for k, v in stats.items():
            hlog.value(k, v / count)
        hlog.value('loss', loss)

