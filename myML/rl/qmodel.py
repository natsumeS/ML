from myML.rl.estimator import QFunc
from typing import Callable
import numpy as np


class QLearning(QFunc):
    def update(self, state, reward, done, *, alpha=0.2):
        # calculate td error
        td_error = reward
        if not done:
            td_error += self.gamma * np.max(self.predict(state))
        td_error -= self.get_value(self.learn_tmp_state, self.learn_tmp_action)

        # update q value
        self.add_value(self.learn_tmp_state, self.learn_tmp_action, alpha * td_error)


class Sarsa(QFunc):

    def __init__(self, num_state: int, num_action: int, get_state_id_func: Callable[..., int], *, gamma=0.99):
        super().__init__(num_state, num_action, get_state_id_func, gamma=gamma)
        self.learn_old_state = None
        self.learn_old_action = None
        self.reward = None

    def reset_learn(self):
        super().reset_learn()
        self.learn_old_state = None
        self.learn_old_action = None
        self.reward = None

    def get_eps_greedy_action(self, state, *, eps=0.1) -> int:
        self.learn_old_state = self.learn_tmp_state
        self.learn_old_action = self.learn_tmp_action
        return super().get_eps_greedy_action(state, eps=eps)

    def get_softmax_action(self, state, *, theta=0.2) -> int:
        self.learn_old_state = self.learn_tmp_state
        self.learn_old_action = self.learn_tmp_action
        return super().get_softmax_action(state, theta=theta)

    def update(self, reward, done, *, alpha=0.2):
        # we know next_action?
        if self.learn_old_state is None:
            if done:
                # calculate td error and update q value of first state
                self.done_end_update(reward, alpha)
                self.reset_learn()
                return
            self.reward = reward
            return

        # calculate td error
        td_error = self.reward + self.gamma * self.get_value(self.learn_tmp_state, self.learn_tmp_action)
        td_error -= self.get_value(self.learn_old_state, self.learn_old_action)

        # update q value
        self.add_value(self.learn_old_state, self.learn_old_action, alpha * td_error)

        # reach the goal state?
        if done:
            self.done_end_update(reward, alpha)
            self.reset_learn()
            return
        self.reward = reward

    def done_end_update(self, reward, alpha=0.2):
        td_error = reward - self.get_value(self.learn_tmp_state, self.learn_tmp_action)
        self.add_value(self.learn_tmp_state, self.learn_tmp_action, alpha * td_error)


class ActorCritic(QFunc):
    def __init__(self, num_state: int, num_action: int, get_state_id_func: Callable[..., int], *, gamma=0.99):
        super().__init__(num_state, num_action, get_state_id_func, gamma=gamma)
        self.value_func = np.zeros(num_state)

    def get_action(self, state) -> int:
        return super().get_softmax_action(state, theta=1.0)

    def update(self, state, reward, done, *, alpha=0.2, beta=0.1):
        # calculate td error by Value func
        td_error = reward
        if not done:
            td_error += self.gamma * self.value_func[self.get_state_id_func(state)]
        td_error -= self.value_func[self.get_state_id_func(self.learn_tmp_state)]

        # update q and v value
        self.value_func[self.get_state_id_func(self.learn_tmp_state)] += alpha * td_error
        self.add_value(self.learn_tmp_state, self.learn_tmp_action, beta * td_error)

