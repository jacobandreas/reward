import gflags
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

FLAGS = gflags.FLAGS
gflags.DEFINE_float('lr', None, 'learning rate')
gflags.DEFINE_float('discount', None, 'discount factor')
gflags.DEFINE_float('w_value', None, 'value loss weight')
gflags.DEFINE_float('w_entropy', None, 'entropy reg weight')

N_HIDDEN = 64

class RnnModel(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.rep = nn.Sequential(
            nn.Linear(n_obs, N_HIDDEN),
            nn.Tanh(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            )

        self.rnn = nn.GRU(
            input_size=N_HIDDEN,
            hidden_size=N_HIDDEN,
            num_layers=1,
            batch_first=True)

        self.act_logits = nn.Linear(N_HIDDEN, n_act)
        self.value = nn.Linear(N_HIDDEN, 1)

        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.softmax = nn.Softmax(dim=2)
        self.opt = optim.RMSprop(self.parameters(), lr=3e-4, eps=1e-5, alpha=0.99)

        self.e_softmax = nn.Softmax(dim=1)
        self.e_log_softmax = nn.LogSoftmax(dim=1)

        self._rnn_state = None

    def reset(self, batch):
        self._rnn_state = Variable(torch.zeros(1, batch.obs.data.shape[0], N_HIDDEN))
        if next(self.rnn.parameters()).is_cuda:
            self._rnn_state = self._rnn_state.cuda()
        return self._rnn_state[0, ...]

    @profile
    def forward(self, batch):
        n_batch, n_slice = batch.obs.data.shape[:2]
        rep = self.rep(batch.obs)
        rnn_rep, _ = self.rnn(rep, batch.mstate)
        logits = self.act_logits(rnn_rep).view(n_batch * n_slice, -1)
        value = self.value(rnn_rep).view(n_batch * n_slice)
        adv = batch.rew.view(n_batch * n_slice) - value
        surrogate = self.loss(logits, batch.act.view(n_batch * n_slice)) * adv.detach()
        neg_entropy = (self.e_softmax(logits) * self.e_log_softmax(logits)).sum(dim=1)
        loss = (
            surrogate.mean() 
            + FLAGS.w_value * adv.pow(2).mean()
            + FLAGS.w_entropy * neg_entropy.mean())
        return loss

    def train(self, batch):
        loss = self(batch)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.data.cpu().numpy()[0])

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
        # strip stack dimension
        model_state = self._rnn_state.data[0, ...]
        return actions, model_state
