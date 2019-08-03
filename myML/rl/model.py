import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from abc import ABCMeta, abstractmethod
from myML.rl.policy import Policy, DiscreteSoftMaxPolicy, GaussianPolicy
from typing import Callable


class Model(chainer.Chain, metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_eps_greedy_action(self, state, eps=0.1):
        pass


class PolicyValueModel(Model, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args) -> (Policy, chainer.Variable):
        pass

    def get_policy(self, state):
        return self.__call__(state)[0]

    def get_value(self, state):
        return self.__call__(state)[1]


class DisCreteSoftMaxPolicyValueModel(PolicyValueModel):
    def __init__(self, num_share_model_output: int, num_action: int):
        super().__init__()
        with self.init_scope():
            self.policy = L.Linear(num_share_model_output, num_action)
            self.value = L.Linear(num_share_model_output, 1)
        self.num_action = num_action

    def __call__(self, x):
        return DiscreteSoftMaxPolicy(self.policy(x), self.num_action), self.value(x)

    def get_action(self, state) -> int:
        with chainer.no_backprop_mode():
            return self.__call__([state])[0].get_action()

    def get_eps_greedy_action(self, state, eps=0.1):
        if np.random.sample() < eps:
            return np.random.randint(self.num_action)
        return self.get_action(state)

    def get_num_action(self):
        return self.num_action


class GaussianPolicyValueModel(PolicyValueModel):
    def __init__(self, num_share_model_output: int, dim_action: int):
        super().__init__()
        with self.init_scope():
            self.mean = L.Linear(num_share_model_output, dim_action)
            self.var = L.Linear(num_share_model_output, dim_action)
            self.value = L.Linear(num_share_model_output, 1)
        self.dim_action = dim_action

    def __call__(self, x):
        return GaussianPolicy(self.mean(x), self.var(x)), self.value(x)

    def get_action(self, state):
        return self.__call__([state])[0].get_action()

    def get_eps_greedy_action(self, state, eps=0.1):
        if np.random.sample() < eps:
            return np.random.uniform(size=self.dim_action)
        return self.get_action(state)
