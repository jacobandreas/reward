from misc import hlog
from rl.data import Batch, Stats, Transition

from collections import Counter
import gflags
import itertools
from joblib import Parallel, delayed
import numpy as np
import torch
from torch import optim

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean('gpu', False, 'use the gpu')
gflags.DEFINE_float('discount', None, 'discount factor')
gflags.DEFINE_float('lr', None, 'learning rate')
gflags.DEFINE_integer('iters', 10000, 'number of iterations to run for')

# TODO magic
N_BATCH = 500
#N_SLICE = 100
#N_ROLLOUT = 50
N_SLICE = 20
N_ROLLOUT = 20

class EnvWrapper(object):
    def __init__(self, env_builder):
        self._underlying = env_builder()

    def reset(self):
        return self._underlying.reset()

    def step(self, action):
        if self._underlying.done():
            return (None, None, None)
        return self._underlying.step(action)

@profile
def _rollout(model, envs, init_mstate=None):
    steps = [e.reset() for e in envs]
    obs, rew, term = zip(*steps)

    batch = Batch.from_obs(obs)
    model_state = model.reset(batch, init_mstate)
    last_mstate = [None for _ in range(len(obs))]

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
        #print("*", act)
        steps = [e.step(a) for e, a in zip(envs, act)]
        obs_, rew, term = zip(*steps)
        for i in range(len(obs_)):
            if done[i]:
                continue
            done[i] = done[i] or term[i]
            if term[i]:
                last_mstate[i] = model_state_[i]
            bufs[i].append(Transition(obs[i], model_state[i].data, act[i], rew[i], None))
        obs = [o_ if o_ is not None else o for o, o_ in zip(obs, obs_)]
        model_state = model_state_

    for i in range(len(obs)):
        if last_mstate[i] is None:
            last_mstate[i] = model_state[i]

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
        forward_bufs.append(forward_buf)
        r += (sum(t.reward for t in buf))

    return forward_bufs, stats, torch.stack(last_mstate)

def _rollout_meta(model1, model2, envs, meta_featurizer):
    feat_envs = [EnvWrapper(lambda: meta_featurizer(e._underlying)) for e in envs]
    bufs1, _, mstate = _rollout(model1, feat_envs)
    bufs2, stats, mstate2 = _rollout(model2, envs, mstate)

    aug_bufs1 = []
    for i in range(len(bufs2)):
        last_fr = bufs2[i][0].forward_reward
        aug_buf = [
            transition._replace(forward_reward=last_fr)
            for transition in bufs1[i]]
        aug_bufs1.append(aug_buf)
    return (aug_bufs1, bufs2), stats, mstate2

@profile
def _train(rollout_fn, train_fn, train_env_builder, val_env_builder, cache_file):
    stats = Stats.empty()
    loss = 0
    # TODO lots of magic
    val_envs = [EnvWrapper(val_env_builder) for _ in range(20)]
    for i_iter in hlog.loop('iter_%05d', range(FLAGS.iters)):
        bufs = None
        count = 0
        while count < N_BATCH:
            envs = [EnvWrapper(train_env_builder) for _ in range(20)]
            rbuf, rstats, _ = rollout_fn(envs)
            if bufs is None:
                bufs = [[] for _ in rbuf]
            for b, rb in zip(bufs, rbuf):
                b += rb
            count += len(rbuf[0])
            stats.update(rstats)
        batches = Batch.from_bufs(bufs, N_BATCH, N_SLICE)
        if FLAGS.gpu:
            batches = [b.cuda() for b in batches]
        loss += train_fn(*batches)

        if (i_iter + 1) % 10 == 0:
            for k, v in stats.items():
                hlog.value(k, v)
            hlog.value('loss', loss)
            stats = Stats.empty()
            loss = 0

            with hlog.task('val'):
                vstats = Stats.empty()
                for _ in range(10):
                    _, vrstats, _ = rollout_fn(val_envs)
                    vstats.update(vrstats)

                for k, v in vstats.items():
                    hlog.value(k, v)

# TODO some dup
def train(model, train_env_builder, val_env_builder, cache_file):
    opt = optim.RMSprop(
        itertools.chain(model1.parameters(), model2.parameters()),
        lr=FLAGS.lr,
        eps=1e-5,
        alpha=0.99)

    if FLAGS.gpu:
        model.cuda()

    def train_helper(batch):
        loss, _ = model(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.data.cpu().numpy()[0])

    _train(
        lambda envs: _rollout(model, envs),
        train_helper, train_env_builder, val_env_builder, cache_file)

def train_meta(model1, model2, train_env_builder, val_env_builder, meta_featurizer, cache_file):
    opt = optim.RMSprop(
        itertools.chain(model1.parameters(), model2.parameters()),
        lr=FLAGS.lr,
        eps=1e-5,
        alpha=0.99)

    if FLAGS.gpu:
        model1.cuda()
        model2.cuda()

    def train_helper(batch1, batch2):
        loss1, mstate1 = model1(batch1)
        loss2, _ = model2(batch2, mstate1)
        loss = loss1 + loss2
        opt.zero_grad()
        loss.backward()
        opt.step()
        return np.asarray(
            [float(l.data.cpu().numpy()[0]) for l in [loss1, loss2]])

    _train(
        lambda envs: _rollout_meta(model1, model2, envs, meta_featurizer),
        train_helper, train_env_builder, val_env_builder, cache_file)
