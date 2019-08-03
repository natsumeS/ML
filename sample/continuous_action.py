import gym
import numpy as np
import chainer.links as L
import chainer.functions as F
import chainer
from myML.rl.model import GaussianPolicyValueModel
from myML.rl.ppo import PPO
from myML.rl.a3c import A3C


class myPVModel(GaussianPolicyValueModel):
    def __init__(self):
        super().__init__(10, 1)
        with self.init_scope():
            self.l1 = L.Linear(2, 10)

    def __call__(self, x):
        x = np.array(x).astype(np.float32)
        h = F.relu(self.l1(x))
        return super().__call__(h)


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")


    def make_env():
        return gym.make("MountainCarContinuous-v0")

    model = myPVModel()
    ppo = PPO(model, make_env)
    ppo.start(8)
    # a3c=A3C(model,make_env)
    # a3c.start(8,2)

    for episode in range(5):
        state = env.reset()
        episode_reward=0.0
        env.render()
        for t in range(200):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            episode_reward+=reward
            env.render()
            if done:
                break
        print("episode_reward={}".format(episode_reward))
