import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from abc import ABCMeta, abstractmethod
from typing import Callable


class Model(chainer.Chain, metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_eps_greedy_action(self, state, eps=0.1):
        pass


class DisCreteSoftMaxPolicyValueModel(Model):
    def __init__(self, num_share_model_output: int, num_action: int):
        super().__init__()
        with self.init_scope():
            self.policy = L.Linear(num_share_model_output, num_action)
            self.value = L.Linear(num_share_model_output, 1)
        self.num_action = num_action

    def __call__(self, x):
        return F.softmax(self.policy(x)), self.value(x)

    def get_policy(self, state) -> chainer.Variable:
        return self.__call__(state)[0]

    def get_action(self, state) -> int:
        with chainer.no_backprop_mode():
            return np.random.choice(self.num_action, size=1, p=self.get_policy([state]).data[0])[0]

    def get_eps_greedy_action(self, state, eps=0.1):
        if np.random.sample() < eps:
            return np.random.randint(self.num_action)
        return self.get_action(state)

    def get_num_action(self):
        return self.num_action
