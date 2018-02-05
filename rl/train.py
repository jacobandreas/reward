from misc import hlog
from rl.data import Batch, Stats, Transition

from collections import Counter
import gflags
from joblib import Parallel, delayed
import numpy as np

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean('gpu', False, 'use the gpu')

# TODO magic
N_BATCH = 1000
N_SLICE = 5

class EnvWrapper(object):
    def __init__(self, env_builder):
        self.underlying = env_builder()

    def reset(self):
        return self.underlying.reset()

    def step(self, action):
        if self.underlying.done():
            return (None, None, None)
        return self.underlying.step(action)

@profile
def rollout(model, envs):
    steps = [e.reset() for e in envs]
    obs, rew, term = zip(*steps)
    batch = Batch.from_obs(np.asarray(obs))
    model_state = model.reset(batch)

    done = list(term)
    bufs = [[] for _ in envs]
    # TODO magic
    for t in range(50):
        if all(done):
            break
        if FLAGS.gpu:
            batch = batch.cuda()
        act, model_state_ = model.act(batch)
        steps = [e.step(a) for e, a in zip(envs, act)]
        obs_, rew, term = zip(*steps)
        for i in range(len(obs_)):
            if done[i]:
                continue
            done[i] = done[i] or term[i]
            bufs[i].append(Transition(obs[i], model_state[i], act[i], rew[i], None))
        obs = [o_ if o_ is not None else o for o, o_ in zip(obs, obs_)]
        model_state = model_state_

    mc_bufs = []
    stats = Stats.empty()
    for buf in bufs:
        mc_rew = 0
        mc_buf = []
        stats.update(Stats.of(buf))
        for transition in buf[::-1]:
            # TODO magic
            mc_rew = mc_rew * 0.95 + transition.reward
            mc_buf.append(transition._replace(forward_reward=mc_rew))
        mc_bufs.append(mc_buf)

    return mc_bufs, stats

@profile
def train(model, env_builder, cache_file):
    if FLAGS.gpu:
        model = model.cuda()
    # TODO magic
    envs = [EnvWrapper(env_builder) for _ in range(10)]
    stats = Stats.empty()
    loss = 0
    for i_iter in hlog.loop('iter_%05d', range(1000)):
        bufs = []
        count = 0
        while count < 2 * N_BATCH:
            rbuf, rstats = rollout(model, envs)
            bufs += rbuf
            count += sum(len(b) for b in rbuf)
            stats.update(rstats)
        batch = Batch.from_bufs(bufs, N_BATCH, N_SLICE)
        if FLAGS.gpu:
            batch = batch.cuda()
        loss += model.train(batch)

        if (i_iter + 1) % 10 == 0:
            for k, v in stats.items():
                hlog.value(k, v)
            hlog.value('loss', loss)
            stats = Stats.empty()
            loss = 0

    #eval_bufs, _ = rollout(model, envs)
    #best = max(eval_bufs, key=lambda b: sum(r for o, a, r, mr in b))
    #actions = [a for o, a, r, mr in best]
    #hlog.value('best_rew', sum(r for o, a, r, mr in b))
    #with open(cache_file, 'w') as cache_f:
    #    print(' '.join(str(a) for a in actions), file=cache_f)
