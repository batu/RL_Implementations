import gym
import numpy as np

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Q_table = np.zeros([env.observation_space.n,
                    env.action_space.n])
#sometimes alpha
learning_rate = .99
#sometimes gamma
discount_rate = .99

num_episodes = 15000




dim_learning_rate = Real(low=.0, high= 1,
                         name='learning_rate')
dim_discount_rate = Real(low=.0, high= 1,
                         name='discount_rate')

dimension = [dim_learning_rate, dim_discount_rate]
default_parameters = [learning_rate, discount_rate]

@use_named_args(dimensions=dimension)
def fitness(learning_rate, discount_rate):
    rewardsList = []
    for current_epoch in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        max_step_in_episode = 100
        while step < max_step_in_episode:
            step += 1

            # Noise gets smaller and smaller as epochs increase
            noise = np.random.randn(1, env.action_space.n) * (1. / (current_epoch + 1))

            actions_in_state = Q_table[state, :]
            action = np.argmax(actions_in_state + noise)

            next_state, reward, done, _ = env.step(action)
            Q_table[state, action] = learning_rate * (reward + discount_rate * np.max(Q_table[next_state, :]))

            episode_reward += reward
            state = next_state
            if done:
                break

        rewardsList.append(episode_reward)
    print(f"Percent of succesful episodes: {100*sum(rewardsList[-100:])/(100)}%")
    return -(float(sum(rewardsList)) / 100)

search_result = gp_minimize(func=fitness,
                            dimensions=dimension,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

print(search_result.x)
ax = plot_convergence(search_result)
plt.show()
