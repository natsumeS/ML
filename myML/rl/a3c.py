import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
import threading
import time

from myML.rl.model import DisCreteSoftMaxPolicyValueModel
from myML.rl.learner import AsyncLearner


class A3CLearner(AsyncLearner):
    def update_model(self):
        # get learning data
        with self.lock:
            if not self.is_adequate_num_data():
                time.sleep(0)
                return
            states, actions, advantages = self.get_train_buffer()

        # get policy and value
        policies, values = self.model(states)
        # calculate loss
        loss_v = F.squared_error(values, advantages)
        loss_pi = (advantages - values.data) * F.reshape(F.log(
            F.select_item(policies, actions) + 1.0e-10), (len(states), 1))
        loss_ent = F.sum(policies * F.log(policies + 1.0e-10), axis=1, keepdims=True)

        loss = F.mean(loss_v * 0.5 - loss_pi + 0.01 * loss_ent)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()


class A3C:
    def __init__(self, model: DisCreteSoftMaxPolicyValueModel, make_env_func=None, *,
                 lr=1e-3, batch_size=32, gamma=0.99, t_max=8, num_episode=200, num_steps_per_episode=200,
                 eps_start=0.4, eps_end=0.15, eps_steps=75000):
        if make_env_func is None:
            raise Exception("set make_env_func:Callable")
        self.make_env_func = make_env_func

        # share learner
        self.learner = A3CLearner(model, chainer.optimizers.RMSprop(lr=lr))

        # setting
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.t_max = t_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_episode = num_episode
        self.num_steps_per_episode = num_steps_per_episode

        # flag
        self.on_update = False

    def async_update(self, learner: A3CLearner):
        while self.on_update:
            learner.update_model()

    def async_explore(self, learner: A3CLearner):
        # make env
        env = self.make_env_func()

        # individual model
        model = learner.copy_model()

        for episode in range(self.num_episode):
            episode_reward = 0.0
            state = env.reset()
            sar_que = []
            for t in range(1, self.num_steps_per_episode):
                eps = max(self.eps_start - (self.eps_start - self.eps_end) * learner.step / self.eps_steps,
                          self.eps_end)

                action = model.get_eps_greedy_action(state, eps)

                next_state, reward, done, _ = env.step(action)

                sar_que.append([state, action, reward])

                learner.step += 1
                episode_reward += reward

                if t % self.t_max == 0 or done:
                    # calculate and push advantage type train data by backward
                    R = 0.0 if done else model([state])[1].data[0][0]

                    for s, a, r in reversed(sar_que):
                        R = r + self.gamma * R
                        with learner.lock:
                            learner.push_train_buffer(s, a, [R])

                    # sync model
                    model = learner.copy_model()
                    sar_que = []
                if done:
                    print("episode={}:score={}".format(episode, episode_reward))
                    break
                state = next_state

    def start(self, num_explorer: int, num_updaters: int):
        updaters = [threading.Thread(target=self.async_update, args=(self.learner,)) for i in range(num_updaters)]
        explorers = [threading.Thread(target=self.async_explore, args=(self.learner,)) for i in range(num_explorer)]

        self.on_update = True

        # start Thread
        for explorer in explorers:
            explorer.start()
        for updater in updaters:
            updater.start()

        # wait end
        for explorer in explorers:
            explorer.join()

        # stop updater
        self.on_update = False
        for updater in updaters:
            updater.join()


if __name__ == '__main__':
    import gym


    class myPVModel(DisCreteSoftMaxPolicyValueModel):
        def __init__(self, num_hidden: int, num_action: int):
            super().__init__(num_hidden, num_action)
            with self.init_scope():
                self.l1 = L.Linear(4, num_hidden)
                self.l2 = L.Linear(num_hidden, num_hidden)

        def __call__(self, x):
            x = np.array(x).astype(np.float32)
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))
            return super().__call__(h)


    def make_env():
        return gym.make("CartPole-v0")


    a = A3C(myPVModel(10, 2), make_env, lr=5e-3)
    a.start(8, 2)
