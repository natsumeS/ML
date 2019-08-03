import numpy as np
import chainer
import chainer.functions as F
from abc import ABCMeta, abstractmethod


class Policy(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def get_prob(self, action):
        pass

    @abstractmethod
    def get_log_prob(self, action):
        pass

    @abstractmethod
    def entropy(self):
        pass





class DiscreteSoftMaxPolicy(Policy):
    def __init__(self, variable: chainer.Variable, num_action: int):
        self.policy = F.softmax(variable)
        self.num_action = num_action

    def get_action(self):
        return np.random.choice(self.num_action, size=1, p=self.policy.data[0])[0]

    def get_prob(self, action):
        return F.reshape(F.select_item(self.policy, action), (len(action), 1))

    def get_log_prob(self, action):
        return F.log(self.get_prob(action) + 1.0e-10)

    def entropy(self):
        return -F.sum(self.policy * F.log(self.policy + 1.0e-10), axis=1, keepdims=True)


class GaussianPolicy(Policy):
    def __init__(self, mean: chainer.Variable, variance: chainer.Variable):
        self.mean = mean
        # variance should be positive number, so operate ReLU function into variance
        self.var = F.relu(variance) + 1.0e-10

    def get_action(self):
        return F.gaussian(self.mean, F.log(self.var)).data[0]

    def get_prob(self, action):
        return F.exp(self.get_log_prob(action))

    def get_log_prob(self, action):
        return - 0.5 * F.square(action - self.mean) / self.var - F.log(2.0 * np.pi * self.var)

    def entropy(self):
        return 0.5 * F.log(2.0 * np.pi * np.e * self.var)
