import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
import threading

from myML.rl.model import DisCreteSoftMaxPolicyValueModel
from myML.rl.learner import AsyncLearner


class PPOLearner(AsyncLearner):
    def __init__(self, model: DisCreteSoftMaxPolicyValueModel, optimizer: chainer.Optimizer, *, batch_size=32,
                 num_train_per_episode=15, eps=0.2):
        super().__init__(model, optimizer, batch_size=batch_size)
        self.old_model = self.copy_model()
        self.num_train_per_episode = num_train_per_episode
        self.eps = eps

    def update_model(self):
        # start minibatch learning
        for t in range(self.num_train_per_episode):
            # get learning data
            with self.lock:
                states, actions, advantages = self.get_sample_from_train_buffer()

            # get policy and value
            policies, values = self.model(states)
            old_policies, _ = self.old_model(states)

            # calculate loss
            loss_v = F.squared_error(values, np.array(advantages).astype(np.float32))
            loss_ent = F.sum(policies * F.log(policies + 1.0e-10), axis=1, keepdims=True)

            # r = F.select_item(policies, actions) / F.select_item(old_policies, actions)
            r = (F.select_item(policies, actions) + 1.0e-10) / (F.select_item(old_policies, actions) + 1.0e-10)
            r = F.reshape(r, (len(states), 1))
            loss_clip = (advantages - values.data) * F.minimum(r, F.clip(r, 1.0 - self.eps, 1.0 + self.eps))

            loss = F.mean(-loss_clip + loss_v * 0.2 + 0.01 * loss_ent)

            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()
        # update old model
        self.old_model = self.copy_model()


class PPO:
    def __init__(self, model: DisCreteSoftMaxPolicyValueModel, make_env_func=None, *,
                 lr=1e-3, batch_size=32, gamma=0.99, lam=0.95, t_max=8, clip_eps=0.2, num_episode=200,
                 num_steps_per_episode=200,
                 eps_start=0.4, eps_end=0.15, eps_steps=75000, num_train_per_episode=15):
        if make_env_func is None:
            raise Exception("set make_env_func:Callable")
        self.make_env_func = make_env_func

        # share learner
        self.learner = PPOLearner(model, chainer.optimizers.RMSprop(lr=lr), batch_size=batch_size,
                                  num_train_per_episode=num_train_per_episode, eps=clip_eps)

        # setting
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.t_max = t_max
        self.gamma = gamma
        self.lam = lam
        self.num_episode = num_episode
        self.num_steps_per_episode = num_steps_per_episode

        # flag
        self.on_explore = False

    def async_explore(self, learner: PPOLearner, explorer_event: threading.Event, learner_event: threading.Event):
        # make env
        env = self.make_env_func()

        # individual model
        model = learner.copy_model()
        # wait updating model
        explorer_event.wait()
        explorer_event.clear()
        while self.on_explore:
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
                    # calculate generalize advantage estimation
                    # A_t=sum_{i=0}^{T-t}(gamma*lambda)^i * delta_{t+i}
                    # delta_{t}:=gamma * V(s_{t+1}) + r_t - V(s_t)
                    # lambda : [0,1]
                    # in PPO, lambda=0.95

                    # here calculate A_t + V(s_t)
                    R = 0.0
                    if not done:
                        R += self.gamma * model([state])[1].data[0][0]

                    for s, a, r in reversed(sar_que):
                        R += r
                        v = model([s])[1]
                        with learner.lock:
                            learner.push_train_buffer(s, a, [R])
                        R *= self.gamma * self.lam
                        R += self.gamma * (1 - self.lam) * v.data[0][0]

                    sar_que = []

                    # start updateding model
                    learner_event.set()

                    # wait updating model
                    explorer_event.wait()
                    if self.on_explore:
                        explorer_event.clear()
                    # sync model
                    model = learner.copy_model()

                if done:
                    break
                state = next_state

    def start(self, num_explorer: int):
        # event to controll thread
        explorer_event = threading.Event()
        learner_events = [threading.Event() for i in range(num_explorer)]

        # evaluate env
        eval_env = self.make_env_func()
        explorers = [threading.Thread(target=self.async_explore, args=(self.learner, explorer_event, learner_events[i]))
                     for i in range(num_explorer)]

        self.on_explore = True
        for explore in explorers:
            explore.start()

        for episode in range(self.num_episode):
            # restart explorer
            explorer_event.set()

            # wait explorer process
            for learner_event in learner_events:
                learner_event.wait()
                learner_event.clear()
            # start learner process
            self.learner.update_model()

            # evalurate
            model = self.learner.get_model()
            episode_reward = 0.0
            for e in range(2):
                s = eval_env.reset()
                for t in range(200):
                    a = model.get_action(s)
                    s, r, d, _ = eval_env.step(a)
                    episode_reward += r
                    if d:
                        break
            episode_reward /= 2
            print("episode={}:score={}".format(episode, episode_reward))

        self.on_explore = False
        explorer_event.set()
        for explore in explorers:
            explore.join()


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


    a = PPO(myPVModel(10, 2), make_env, lr=5e-3)
    a.start(8)
