import tensorflow as tf
import numpy as np
import gym
from keras import utils as np_utils
import matplotlib.pyplot as plt
from collections import deque
import random

class DQN_Agent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_counter = 0

        self.batch_size = 32


    def create_model(self):
        inputs = tf.keras.Input(shape=self.env.observation_space.shape)
        hidden = tf.keras.layers.Dense(24, activation=tf.nn.relu)(inputs)
        hidden = tf.keras.layers.Dense(48, activation=tf.nn.relu)(inputs)
        hidden = tf.keras.layers.Dense(24, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(self.env.action_space.n)(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy'])

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            current_q_values = self.target_model.predict(np.expand_dims(state, 0))
            if done:
                current_q_values[0][action] = reward
            else:
                future_q_values =self.target_model.predict(np.expand_dims(next_state, 0))
                best_value = max(future_q_values[0])
                current_q_values[0][action] = reward + self.gamma * best_value
            self.model.fit(np.expand_dims(state, 0), current_q_values, epochs=1, verbose=0)


    def update_target(self):
        weights = self.model.get_weights()
        if self.tau == 0:
            self.target_model.set_weights(weights)
            self.update_counter = 0;
        else:
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(np.expand_dims(state, 0)))

env = gym.make("CartPole-v0")
print(env.observation_space)
print(env.action_space)
agent = DQN_Agent(env)

episodes  = 5000
episodes_len = 200
update_target_network = 200

for episode in range(episodes):
    cur_state = env.reset()
    cumulative_reward = 0
    for step in range(episodes_len):
        action = agent.act(cur_state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        agent.remember(cur_state, action, reward, next_state, done)
        agent.replay()
        agent.update_target()

        cur_state = next_state

        if done:
            break
    # print(f"Finsihed episode:{episode} with {'*********** success ***********.' if reward > 0 else 'failure.'}")
    print(f"Finsihed episode:{episode} with reward: {cumulative_reward} and epsilon:{agent.epsilon:.3f}")
