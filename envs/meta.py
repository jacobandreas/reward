import numpy as np

class MetaFeatureEnv(object):
    def __init__(self, underlying):
        self._underlying = underlying
        self.n_actions = underlying.n_actions
        self.feature_shape = underlying.feature_shape + ((self.n_actions + 2,),)
        self.featurizer = underlying.featurizer.extend(self.feature_shape)

    def reset(self):
        obs, rew, term = self._underlying.reset()
        extra_features = np.concatenate((np.zeros(self.n_actions), [0, 0]))
        return obs + (extra_features,), rew, term

    def step(self, action):
        obs, rew, term = self._underlying.step(action)
        act_features = np.zeros(self.n_actions)
        act_features[action] = 1
        extra_features = np.concatenate((act_features, [rew, term]))
        return obs + (extra_features,), rew, term

    def done(self):
        return self._underlying.done()

# TODO duplication
class MetaRlWrapperEnv(object):
    def __init__(self, underlying):
        self._underlying = underlying
        self._phase = None
        self.n_actions = underlying.n_actions
        self.feature_shape = underlying.feature_shape + ((self.n_actions + 2,),)
        self.featurizer = underlying.featurizer.extend(self.feature_shape)

    def reset(self):
        self._phase = 0
        obs, rew, term = self._underlying.reset()
        assert rew == 0
        assert not term
        extra_features = np.concatenate((
            np.zeros(self.n_actions),
            [0, 0]))
        full_obs = obs + (extra_features,)
        return full_obs, 0, False 

    def step(self, action):
        assert self._phase is not None
        obs, rew, term = self._underlying.step(action)
        true_term = term and self._phase == 1
        true_rew = rew if self._phase == 1 else 0

        if term:
            if self._phase == 0:
                obs2, rew2, term2 = self._underlying.reset()
                assert rew2 == 0
                assert not term2
                obs = obs2
                self._phase = 1
            elif self._phase == 1:
                self._phase = None

        act_features = np.zeros(self.n_actions)
        act_features[action] = 1
        extra_features = np.concatenate((
            act_features,
            [rew, term]))

        full_obs = obs + (extra_features,)

        return full_obs, true_rew, true_term

    def done(self):
        return self._underlying.done()
