import numpy as np

from typing import Callable
import chainer
import chainer.links as L
import chainer.functions as F

import threading
import time


class PolicyValueModel(chainer.Chain):
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
            return np.random.choice(self.num_action, size=1, p=self.get_policy(state).data[0])[0]

    def get_eps_greedy_action(self, state, *, eps=0.1):
        if np.random.sample() < eps:
            return np.random.randint(self.num_action)
        return self.get_action(state)

    def get_num_action(self):
        return self.num_action


class A3COptimizerAgent(threading.Thread):
    is_end = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.is_end:
            A3C.update()

    @classmethod
    def terminate(cls):
        cls.is_end = True


class A3CActionAgent(threading.Thread):
    shared_step = 0

    def __init__(self, env, thread_id: int, num_episode: int = 2000, num_steps_per_episode: int = 200, *, gamma=0.99,
                 t_max=8, eps_start=0.4, eps_end=0.15, eps_steps=75000,
                 on_end_episode: Callable[[int, float, PolicyValueModel], None] = None):
        threading.Thread.__init__(self)
        # unique model
        self.model = A3C.sync_model()
        self.env = env
        self.num_action = self.model.get_num_action()
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.t_max = t_max
        self.gamma = gamma
        self.num_episode = num_episode
        self.num_steps_per_episode = num_steps_per_episode
        self.que_sar = []
        self.thread_id = thread_id  # 0,1,2,..:int 0=main thread
        self.on_end_episode = on_end_episode

    def get_action(self, state):
        with chainer.no_backprop_mode():
            return np.random.choice(self.num_action, size=1, p=self.model([state])[0].data[0])[0]

    def get_eps_greedy_action(self, state):
        eps = max(self.eps_start - (self.eps_start - self.eps_end) * self.shared_step / self.eps_steps,
                  self.eps_end)
        if np.random.sample() < eps:
            return np.random.randint(self.num_action)
        return self.get_action(state)

    def push_data(self, state, done):
        R = 0 if done else self.model([state])[1].data[0][0]
        for s, a, r in reversed(self.que_sar):
            # get policy,value and R
            policy, value = self.model([s])
            R = r + self.gamma * R

            A3C.push_train_que(s, a, [R])
        self.que_sar = []

    def run(self):

        for episode in range(self.num_episode):
            episode_reward = 0.0
            state = self.env.reset()
            for t in range(1, self.num_steps_per_episode):
                action = self.get_eps_greedy_action(state)
                que_element = [state, action, None]

                state, reward, done, _ = self.env.step(action)

                que_element[2] = reward
                self.que_sar.append(que_element)
                self.shared_step += 1
                episode_reward += reward
                if t % self.t_max == 0 or done:
                    self.push_data(state, done)
                    self.model = A3C.sync_model()
                if done:
                    if self.thread_id == 0:
                        print("step={},episode={}:R={}".format(self.shared_step, episode, episode_reward))
                        if self.on_end_episode is not None:
                            self.on_end_episode(episode, episode_reward, self.model)
                    break


class A3C:
    model = None
    optimizer = None
    train_que = [[], [], []]
    lock_que = threading.Lock()
    batch_size = 32

    def __init__(self, model: PolicyValueModel, make_env_func=None):
        self.__class__.model = model
        if make_env_func is None:
            raise Exception("set make_env_func:Callable")
        self.make_env_func = make_env_func

    def learning(self, *, lr=1e-3, batch_size=32, gamma=0.99, t_max=8, num_episode=200, num_step_per_episode=200,
                 eps_start=0.4, eps_end=0.15, eps_steps=75000, num_agent=8, num_optimizer=2,
                 on_end_episode: Callable[[int, float, PolicyValueModel], None] = None):
        # set optimizer
        self.__class__.optimizer = chainer.optimizers.RMSprop(lr=lr)
        self.__class__.optimizer.setup(self.__class__.model)
        self.__class__.batch_size = batch_size

        # set ActionAgent
        agents = [A3CActionAgent(self.make_env_func(), tid, num_episode=num_episode,
                                 num_steps_per_episode=num_step_per_episode, gamma=gamma, t_max=t_max,
                                 eps_start=eps_start, eps_end=eps_end,
                                 eps_steps=eps_steps, on_end_episode=on_end_episode) for tid in range(num_agent)]

        # set OptimizerAgent
        opts = [A3COptimizerAgent() for i in range(num_optimizer)]

        for agent in agents:
            agent.start()
        for optimizer in opts:
            optimizer.start()

        for agent in agents:
            agent.join()
        for optimizer in opts:
            optimizer.terminate()

    @classmethod
    def initial_setting(cls, model: chainer.Chain, batch_size=32):
        cls.model = model
        cls.batch_size = batch_size
        cls.optimizer = chainer.optimizers.RMSprop(lr=5e-3)
        cls.optimizer.setup(cls.model)

    @classmethod
    def update(cls):
        with cls.lock_que:
            if len(cls.train_que[0]) < cls.batch_size:
                time.sleep(0)
                return
            states, actions, advantages = cls.train_que
            this_batch_size = len(states)
            cls.train_que = [[], [], []]

        states = np.array(states).astype(np.float32)
        actions = np.array(actions).astype(np.int32)
        advantages = np.array(advantages).astype(np.float32)
        policies, values = cls.model(states)

        loss_v = F.squared_error(values, advantages)
        loss_pi = (advantages - values.data) * F.reshape(F.log(
            F.select_item(policies, actions) + 1.0e-10), (this_batch_size, 1))
        loss_ent = F.sum(policies * F.log(policies + 1.0e-10), axis=1, keepdims=True)

        loss = loss_v * 0.5 - loss_pi + 0.01 * loss_ent
        loss = F.mean(loss)

        cls.model.cleargrads()
        loss.backward()
        cls.optimizer.update()

    @classmethod
    def push_train_que(cls, state, action, advantage):
        with cls.lock_que:
            cls.train_que[0].append(state)
            cls.train_que[1].append(action)
            cls.train_que[2].append(advantage)

    @classmethod
    def sync_model(cls):
        return cls.model.copy(mode='copy')


if __name__ == '__main__':
    import gym

    class myPVModel(PolicyValueModel):
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


    agent = A3C(myPVModel(10, 2), make_env)
    agent.learning(lr=5e-3, num_episode=300)
