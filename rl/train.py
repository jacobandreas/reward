from misc import hlog
from rl.data import Batch, Stats, Transition

from collections import Counter
import gflags
from joblib import Parallel, delayed
import numpy as np

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean('gpu', False, 'use the gpu')
gflags.DEFINE_float('discount', None, 'discount factor')

# TODO magic
N_BATCH = 2000
N_SLICE = 100
N_ROLLOUT = 50

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

    batch = Batch.from_obs(obs)
    model_state = model.reset(batch)

    done = list(term)
    bufs = [[] for _ in envs]
    # TODO magic
    for t in range(N_ROLLOUT):
        if all(done):
            break
        batch = Batch.from_obs(obs)
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

    forward_bufs = []
    stats = Stats.empty()
    r = 0
    for buf in bufs:
        stats.update(Stats.of(buf))
        forward_rew = 0
        forward_rews = []
        for transition in buf[::-1]:
            # TODO magic
            forward_rew = forward_rew * FLAGS.discount + transition.reward
            forward_rews.append(forward_rew)
        forward_rews = list(reversed(forward_rews))
        forward_buf = [
                transition._replace(forward_reward=fr)
                for transition, fr in zip(buf, forward_rews)]
        #forward_bufs.append(list(reversed(forward_buf)))
        forward_bufs.append(forward_buf)
        r += (sum(t.reward for t in buf))

    print(r / len(bufs))
    return forward_bufs, stats

@profile
def train(model, train_env_builder, val_env_builder, cache_file):
    if FLAGS.gpu:
        model = model.cuda()
    # TODO magic
    stats = Stats.empty()
    loss = 0
    val_envs = [EnvWrapper(val_env_builder) for _ in range(20)]
    for i_iter in hlog.loop('iter_%05d', range(10000)):
        bufs = []
        count = 0
        while count < 2 * N_BATCH:
            envs = [EnvWrapper(train_env_builder) for _ in range(20)]
            rbuf, rstats = rollout(model, envs)
            #print()
            #print([t.action for t in rbuf[0]], rbuf[0][-1].reward, count)
            #print([t.forward_reward for t in rbuf[0]])
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

            with hlog.task('val'):
                vstats = Stats.empty()
                for _ in range(10):
                    _, vrstats = rollout(model, val_envs)
                    vstats.update(vrstats)

                for k, v in vstats.items():
                    hlog.value(k, v)

    #eval_bufs, _ = rollout(model, envs)
    #best = max(eval_bufs, key=lambda b: sum(r for o, a, r, mr in b))
    #actions = [a for o, a, r, mr in best]
    #hlog.value('best_rew', sum(r for o, a, r, mr in b))
    #with open(cache_file, 'w') as cache_f:
    #    print(' '.join(str(a) for a in actions), file=cache_f)
