import gym 
import random
import time


class NaiveAgent(object):

    def __init__(self, env):
        self.action_size = env.action_space.n

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action


def main():
    # create gym environment
    #  env_name = "CartPole-v1"
    env_name = "ALE/Breakout-v5"
    env = gym.make(env_name)
    print("Observation Space", env.observation_space)
    print("Action Space: ", env.action_space.n)
        
    # create agent
    agent = NaiveAgent(env)
    state = env.reset()

    # train agent
    for _ in range(200):
        action = agent.get_action(state) #  action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
