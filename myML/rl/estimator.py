import numpy as np
from typing import Callable

class QFunc:
    def __init__(self, num_state: int, num_action: int, get_state_id_func: Callable[..., int], *, gamma=0.99):
        self.num_state = num_state
        self.num_action = num_action
        self.get_state_id_func = get_state_id_func
        self.func = np.zeros((num_state, num_action))
        self.learn_tmp_state = None
        self.learn_tmp_action = None
        self.gamma = gamma

    def predict(self, state) -> [float]:
        return self.func[self.get_state_id_func(state)]

    def get_value(self, state, action) -> float:
        return self.func[self.get_state_id_func(state)][action]

    def get_action(self, state) -> int:
        return np.argmax(self.predict(state))

    def reset_learn(self):
        self.learn_tmp_state = None
        self.learn_tmp_action = None

    def get_eps_greedy_action(self, state, *, eps=0.1) -> int:
        self.learn_tmp_state = state
        if np.random.rand() < eps:
            self.learn_tmp_action = np.random.randint(self.num_action)
        else:
            self.learn_tmp_action = self.get_action(state)
        return self.learn_tmp_action

    def get_softmax_action(self, state, *, theta=0.2) -> int:
        self.learn_tmp_state = state
        p = np.exp(theta * self.predict(state))
        psum = np.sum(p)
        p = p / psum
        self.learn_tmp_action = np.random.choice(np.arange(0, self.num_action), p=p)
        return self.learn_tmp_action

    def add_value(self, state, action, target):
        self.func[self.get_state_id_func(state)][action] += target

    def set_value(self, state, action, target):
        self.func[self.get_state_id_func(state)][action] = target
