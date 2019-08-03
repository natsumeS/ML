import numpy as np
import chainer
from abc import ABCMeta, abstractmethod
import threading, random

from myML.rl.model import Model


class Learner(metaclass=ABCMeta):
    def __init__(self, model: Model, optimizer: chainer.Optimizer, *, batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        # train buffer
        self.train_buffer = None
        self.clear_buffer()
        self.batch_size = batch_size

    @abstractmethod
    def clear_buffer(self):
        pass

    @abstractmethod
    def push_train_buffer(self,*args):
        pass

    @abstractmethod
    def get_data_from_train_buffer(self):
        pass

    # def get_train_buffer(self):
    #     # get train data
    #     states, actions, advantages = self.train_buffer
    #
    #     # clear buffer
    #     self.train_buffer = [[], [], []]
    #
    #     # rearange data
    #     states = np.array(states).astype(np.float32)
    #     actions = np.array(actions).astype(np.int32)
    #     advantages = np.array(advantages).astype(np.float32)
    #
    #     return states, actions, advantages

    def get_model(self):
        return self.model

    def copy_model(self):
        return self.model.copy(mode='copy')

    @abstractmethod
    def update_model(self):
        pass


class AsyncLearner(Learner):
    def __init__(self, model: Model, optimizer: chainer.Optimizer, *, batch_size=32):
        super().__init__(model, optimizer, batch_size=batch_size)
        self._lock = threading.Lock()
        self._step = 0

    @property
    def lock(self):
        return self._lock

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step
