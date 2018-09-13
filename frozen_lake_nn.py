import gym
import numpy as np
import tensorflow as tf

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

import matplotlib.pyplot as plt
import random

# Random agent succesful episodes: 0.014603%
# Percent of succesful episodes: 0.040008001600320066%

# Percent of succesful episodes: 0.130226%

env = gym.make('FrozenLake-v0')

Q_table = np.zeros([env.observation_space.n,
                    env.action_space.n])

#sometimes gamma
discount_rate = 1
epsilon = .1
num_episodes = 10000

dim_discount_rate = Real(low=.0, high= 1,
                         name='discount_rate')

dimension = [dim_discount_rate]
default_parameters = [discount_rate]

tf.reset_default_graph()

inputs = tf.keras.Input(shape=(16,))
hidden = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(4)(hidden)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

def one_hot_state(state_index:int)->list:
    state = np.zeros(16)
    state[state_index] = 1
    return np.array((state,))

@use_named_args(    dimensions=dimension)
def fitness(discount_rate):
    global epsilon
    rewardsList = [0] * 100
    for current_epoch in range(num_episodes):
        state = one_hot_state(env.reset())
        episode_reward = 0
        done = False

        step = 0
        max_step_in_episode = 100
        while step < max_step_in_episode:
            step += 1
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                act_values = model.predict(state)
                action = np.argmax(act_values)

            next_state_index, reward, done, _ = env.step(action)
            next_state = one_hot_state(next_state_index)
            target = reward + discount_rate * np.max(model.predict(next_state))

            target_f = model.predict(state)
            target_f[0][action] = target
            # Train the Neural Net with the state and target_f
            history = model.fit(state, target_f, epochs=1, verbose=0)

            episode_reward += reward
            state = next_state

            if done:
                epsilon = 1./((current_epoch/50.) + 10)
                break

        print(f"Episode: {current_epoch}: {episode_reward} Percent of succesful episodes: {100*sum(rewardsList[-100:])/(100)}%")
        rewardsList.append(episode_reward)
    print(f"Percent of succesful episodes: {100*sum(rewardsList[-500:])/500}%")
    return -(float(sum(rewardsList)) / num_episodes)

search_result = gp_minimize(func=fitness,
                            dimensions=dimension,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

fitness([discount_rate])

# print(search_result.x)
# ax = plot_convergence(search_result)
# plt.show()
