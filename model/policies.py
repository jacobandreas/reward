import gflags
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

FLAGS = gflags.FLAGS
gflags.DEFINE_float('w_value', None, 'value loss weight')
gflags.DEFINE_float('w_entropy', None, 'entropy reg weight')

N_HIDDEN = 64

class Entropy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.log_softmax = nn.LogSoftmax(dim=dim)

    def forward(self, logits):
        return -(self.softmax(logits) * self.log_softmax(logits)).sum(dim=1)

# TODO lots of duplication
class SimpleModel(nn.Module):
    def __init__(self, featurizer, n_act):
        super().__init__()
        self.rep = featurizer
        self.rep2 = nn.Sequential(
            nn.Linear(featurizer.n_output + N_HIDDEN, N_HIDDEN),
            nn.Tanh(),
            #nn.Linear(N_HIDDEN, N_HIDDEN),
            #nn.Tanh(),
            )
        self.act_logits = nn.Linear(N_HIDDEN, n_act)
        self.value = nn.Linear(N_HIDDEN, 1)

        self.cross_ent = nn.CrossEntropyLoss(reduce=False)
        self.entropy = Entropy(dim=1)
        self.softmax = nn.Softmax(dim=2)

    def reset(self, batch, init_mstate=None):
        if init_mstate is None:
            init_mstate = Variable(torch.zeros(batch.obs[0].data.shape[0], N_HIDDEN))
            if next(self.rep2.parameters()).is_cuda:
                init_mstate = init_mstate.cuda()
        self._mstate = init_mstate
        return self._mstate

    def forward(self, batch, init_mstate=None, clone=False):
        n_batch, n_slice = batch.obs[0].data.shape[:2]
        if init_mstate is None:
            init_mstate = batch.mstate
        exp_state = init_mstate.unsqueeze(1).expand(n_batch, n_slice, N_HIDDEN)

        rep = self.rep2(torch.cat((self.rep(batch.obs), exp_state), dim=2))
        logits = self.act_logits(rep).view(n_batch * n_slice, -1)
        value = self.value(rep).view(n_batch * n_slice)
        adv = batch.rew.view(n_batch * n_slice) - value
        mask = batch.mask.view(n_batch * n_slice)

        scale = 1 if clone else adv.detach()
        surrogate = self.cross_ent(logits, batch.act.view(n_batch * n_slice)) * scale

        loss = (
            (surrogate * mask).mean()
            + FLAGS.w_value * (adv.pow(2) * mask).mean()
            - FLAGS.w_entropy * (self.entropy(logits) * mask).mean())
        return loss, init_mstate, surrogate

    def act(self, batch):
        v = self._mstate.unsqueeze(1)
        rep = self.rep2(torch.cat(
            (self.rep(batch.obs), self._mstate.unsqueeze(1)),
            dim=2))
        probs = self.softmax(self.act_logits(rep)).data.cpu().numpy()
        # strip time dimension
        probs = probs.squeeze(1)
        actions = np.zeros(probs.shape[0], dtype=np.int32)
        for i, row in enumerate(probs):
            actions[i] = np.random.choice(4, p=row)
        return actions, self._mstate

    def mstate(self):
        return self._mstate

class RnnModel(nn.Module):
    def __init__(self, featurizer, n_act):
        super().__init__()
        self.rep = featurizer

        self.rnn = nn.GRU(
            input_size=featurizer.n_output,
            hidden_size=N_HIDDEN,
            num_layers=1,
            batch_first=True)

        self.act_logits = nn.Linear(N_HIDDEN, n_act)
        self.value = nn.Linear(N_HIDDEN, 1)

        self.cross_ent = nn.CrossEntropyLoss(reduce=False)
        self.entropy = Entropy(dim=1)
        self.softmax = nn.Softmax(dim=2)

        self._rnn_state = None

    def reset(self, batch, init_mstate=None):
        if init_mstate is None:
            init_mstate = Variable(torch.zeros(1, batch.obs[0].data.shape[0], N_HIDDEN))
            if next(self.rnn.parameters()).is_cuda:
                init_mstate = init_mstate.cuda()
        self._rnn_state = init_mstate
        return self._rnn_state[0, ...]

    @profile
    def forward(self, batch, init_mstate=None, extra_reward = 0):
        n_batch, n_slice = batch.obs[0].data.shape[:2]
        rep = self.rep(batch.obs)
        if init_mstate is None:
            init_mstate = batch.mstate
        rnn_rep, _ = self.rnn(rep, init_mstate.unsqueeze(0))

        select = (batch.lens-1).view((-1, 1, 1)).expand(n_batch, 1, N_HIDDEN)
        rnn_state = rnn_rep.gather(1, select).squeeze(1)

        logits = self.act_logits(rnn_rep).view(n_batch * n_slice, -1)
        value = self.value(rnn_rep).view(n_batch * n_slice)
        adv = batch.rew.view(n_batch * n_slice) - value
        mask = batch.mask.view(n_batch * n_slice)
        scale = adv.detach() + extra_reward
        surrogate = self.cross_ent(logits, batch.act.view(n_batch * n_slice)) * scale
        loss = (
            (surrogate * mask).mean()
            + FLAGS.w_value * (adv.pow(2) * mask).mean()
            - FLAGS.w_entropy * (self.entropy(logits) * mask).mean())
        return loss, rnn_state, surrogate

    @profile
    def act(self, batch):
        assert self._rnn_state is not None

        rep = self.rep(batch.obs)
        rnn_rep, self._rnn_state = self.rnn(rep, self._rnn_state)
        probs = self.softmax(self.act_logits(rnn_rep)).data.cpu().numpy()
        # strip time dimension
        probs = probs.squeeze(1)
        actions = np.zeros(probs.shape[0], dtype=np.int32)
        for i, row in enumerate(probs):
            actions[i] = np.random.choice(4, p=row)
            #actions[i] = np.random.choice(2) + 2
        # strip stack dimension
        model_state = self._rnn_state[0, ...]
        return actions, model_state

    def mstate(self):
        return self._rnn_state[0, ...]
