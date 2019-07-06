import numpy as np
from typing import Callable
import keras


class QBaseModel:
    def __init__(self, num_action: int, *, gamma=0.99):
        self.num_action = num_action
        self.model = None
        self.learn_tmp_state = None
        self.learn_tmp_action = None
        self.gamma = gamma

    def predict(self, state) -> [float]:
        raise Exception("override predict method")

    def get_value(self, state, action) -> float:
        return self.predict(state)[action]

    def reset_learn(self):
        self.learn_tmp_state = None
        self.learn_tmp_action = None

    def get_action(self, state) -> int:
        return np.argmax(self.predict(state))

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

    def evaluate(self, env, *, num_episode=5, is_render=False):
        score = 0.0
        for episode in range(num_episode):
            state = env.reset()
            episode_reward = 0.0
            if is_render:
                env.render()
            for t in range(200):
                action = self.get_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if is_render:
                    env.render()
                if done:
                    print("episode={}:{}".format(episode, episode_reward))
                    score += episode_reward
                    break
        return score / num_episode


class QFunc(QBaseModel):
    def __init__(self, num_state: int, num_action: int, get_state_id_func: Callable[..., int], *, gamma=0.99):
        super().__init__(num_action, gamma=gamma)
        self.num_state = num_state
        self.get_state_id_func = get_state_id_func
        self.model = np.zeros((num_state, num_action))

    def predict(self, state) -> [float]:
        return self.model[self.get_state_id_func(state)]

    def add_value(self, state, action, target):
        self.model[self.get_state_id_func(state)][action] += target

    def set_value(self, state, action, target):
        self.model[self.get_state_id_func(state)][action] = target


class QNet(QBaseModel):
    def __init__(self, model: keras.Sequential, num_action: int, *, state_into_input_func=None, gamma=0.99):
        super().__init__(num_action, gamma=gamma)
        self.model = model

    def predict(self, state) -> [float]:
        return self.model.predict(np.array([state]))

    def save_model(self, filename: str, overwrite: bool):
        self.model.save_weights(filename, overwrite=overwrite)

    def load_model(self, filename: str):
        self.model.load_weights(filename)
