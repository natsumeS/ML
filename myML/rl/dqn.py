import numpy as np
import random
from collections import deque
from typing import Callable
import tensorflow as tf
from tensorflow import keras
from myML.helper.plot import PlotHelper
from myML.rl.estimator import QNet


class ExperienceReplayBuffer:
    def __init__(self, maxlen=5000, batch_size=32):
        self.que = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.batch_size = batch_size

    def append(self, state, action, reward, next_state, done):
        # (state, action, reward, next_state, done)
        self.que.append((state, action, reward, next_state, done))

    def sample(self) -> []:
        return random.sample(self.que, self.batch_size)


class DQN(QNet):
    def __init__(self, model: keras.Sequential, num_action: int, *,
                 state_into_input_func=None, gamma=0.99,
                 maxlen=20000, batch_size=32, target_calculate_interval=1, update_network_interval=5,
                 warmup_step=5000):
        super().__init__(model, num_action, gamma=gamma)
        self.model = model
        self.target_model = keras.Sequential(model.layers)
        # replay buffer
        self.replay_buffer = ExperienceReplayBuffer(maxlen=maxlen, batch_size=batch_size)
        self.target_calculate_interval = target_calculate_interval
        self.update_network_interval = update_network_interval
        self.step_counter = 0
        self.warmup_step = warmup_step

        # optimizer and grobal_step
        self.target_model.compile(keras.optimizers.RMSprop(), loss=tf.losses.huber_loss, metrics=['mse'])

    def target_predict(self, state) -> [float]:
        return self.target_model.predict(np.array([state]))

    def update(self, state, reward, done):
        # add replay_buffer
        self.replay_buffer.append(self.learn_tmp_state, self.learn_tmp_action, reward, state, done)

        # update step_counter
        self.step_counter += 1

        # warmup?
        if self.step_counter < self.warmup_step:
            return

        # run gradient descent?
        if self.step_counter % self.target_calculate_interval == 0:
            inputs, targets = self.make_fit_data()
            self.target_model.fit(inputs, targets, verbose=0)

        # update q network?
        if self.step_counter % self.update_network_interval == 0:
            # calculate loss func
            self.model.set_weights(self.target_model.get_weights())

    def make_fit_data(self):
        # sample tuple of states... from replay buffer
        datas = self.replay_buffer.sample()

        # initialize inputs and targets
        inputs = []
        targets = []

        # make inputs and targets from data
        for state, action, reward, next_state, done in datas:
            # add state into inputs list
            inputs.append(state)

            # add target into targets list
            targets.append(self.get_target(state, action, reward, next_state, done))

        # transform numpy list
        return np.array(inputs), np.array(targets)

    def get_target(self, state, action, reward, next_state, done):
        target = self.predict(state)[0]
        target_v = reward
        if not done:
            target_v += self.gamma * np.max(self.predict(next_state))
        target[action] = target_v - target[action]
        return target

    def run(self, env, max_episode: int, eps_func: Callable[[int, int], float], *, output_dir="output",
            output_basefilename="dqn", plot_show=False, is_best_save=False):
        # make plot helper
        ph = PlotHelper(output_dir, output_basefilename, plot_show=plot_show)
        # score
        best_score = None

        for episode in range(max_episode):
            state = env.reset()
            episode_reward = 0
            eps = eps_func(episode, max_episode)
            for t in range(200):
                action = self.get_eps_greedy_action(state, eps=eps)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                self.update(state, reward, done)
                if done:
                    print("episode={}:{}".format(episode, episode_reward))
                    ph.add_data(episode, episode_reward)
                    if is_best_save:
                        score = self.evaluate(env, num_episode=5, is_render=False)
                        if best_score is None or best_score < score:
                            best_score = score
                            self.save_model("{}/{}_best".format(output_dir, output_basefilename), True)
                    break

        # update model weight
        self.model.set_weights(self.target_model.get_weights())


class DoubleDQN(DQN):
    def get_target(self, state, action, reward, next_state, done):
        target = self.target_predict(state)[0]
        target_v = reward
        if not done:
            target_v += self.gamma * np.max(self.target_predict(next_state))
        target[action] = target_v - target[action]
        return target

    def run(self, env, max_episode: int, eps_func: Callable[[int, int], float], *, output_dir="output",
            output_basefilename="double-dqn", plot_show=False, is_best_save=False):
        super().run(env, max_episode, eps_func, output_dir=output_dir, output_basefilename=output_basefilename,
              plot_show=plot_show, is_best_save=is_best_save)


if __name__ == '__main__':
    import gym

    env = gym.make("CartPole-v0")
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(4,), activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(2)
    ])
    # dqn = DQN(model, 2, warmup_step=500, batch_size=32, maxlen=10000, update_network_interval=20)
    ddqn = DoubleDQN(model, 2, warmup_step=500, batch_size=32, maxlen=10000, update_network_interval=20)

    # learning episode
    MAX_EPISODE = 1200


    def get_eps(episode, max_episode):
        return max(0.5 * (1 - episode / max_episode * 10), 0.0)


    ddqn.run(env, MAX_EPISODE, get_eps, output_basefilename="cartpole-dqn", plot_show=True)
