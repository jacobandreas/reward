N_HIDDEN = 64

import numpy as np
from torch import nn, optim

class Model(nn.Module):
    def __init__(self, n_obs):
        super(Model, self).__init__()
        # TODO magic
        self.rep = nn.Sequential(
                nn.Linear(n_obs, N_HIDDEN),
                nn.Tanh())
        self.act_logits = nn.Linear(N_HIDDEN, 4)
        self.value = nn.Linear(N_HIDDEN, 1)

        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.softmax = nn.Softmax(dim=1)
        self.opt = optim.RMSprop(self.parameters(), lr=3e-4, eps=1e-5, alpha=0.99)

    def forward(self, batch):
        rep = self.rep(batch.obs)
        logits = self.act_logits(rep)
        value = self.value(rep).squeeze(1)
        adv = batch.rew - value
        surrogate = self.loss(logits, batch.act) * adv.detach()
        loss = surrogate.mean() + adv.pow(2).mean()
        return loss

    def train(self, batch):
        loss = self(batch)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.data.numpy()[0])

    def act(self, batch):
        probs = self.softmax(self.act_logits(self.rep(batch.obs))).data.numpy()
        actions = np.zeros(probs.shape[0], dtype=np.int32)
        for i, row in enumerate(probs):
            actions[i] = np.random.choice(4, p=row)
        return actions
