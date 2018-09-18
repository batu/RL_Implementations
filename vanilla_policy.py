import tensorflow as tf
import numpy as np
import gym
from keras import utils as np_utils
import matplotlib.pyplot as plt


class VanillaPolicyAgent():

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32]):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_network(input_dim, output_dim, hidden_dims)
        self.__build_train_fn()

    def __build_network(self, input_dim, output_dim, hidden_dims=[32, 32]):
        """Create a base network"""
        input = tf.keras.layers.Input(shape=(input_dim,))
        nn = input
        for h_layer in hidden_dims:
            nn = tf.keras.layers.Dense(h_layer, activation="relu")(nn)

        output = tf.keras.layers.Dense(output_dim, activation="softmax")(nn)

        self.model = tf.keras.Model(inputs=input, outputs=output)


    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        # This is the last layer of our model, the action probability distribution
        action_prob_placeholder = self.model.output

        # The length of the one_hot is num_actions which is the same as output dimension.
        action_onehot_placeholder   = tf.placeholder(shape=(None, self.output_dim), name="action_onehot", dtype=tf.float32)

        # The scalar reward value.
        discount_reward_placeholder = tf.placeholder(shape=(None), name="discount_reward", dtype=tf.float32)

        # ~~~ This part below is to calculate the loss ~~~
        # This gives us the action probability over the one hot representation [0, 1] * [0.4, 0.6] -> [0, 0.6]
        one_hot_action_probilities = action_prob_placeholder * action_onehot_placeholder

        # combine the the one hot action probablilites sum([0, 0.6], [0.7, 0]) -> [0.6, 0.7]
        action_prob = tf.reduce_sum(one_hot_action_probilities, axis=1)

        # take the log, because of the log likelihood   [0.6, 0.7] -> [-0.6, -0.5]
        log_action_prob = tf.log(action_prob)

        # log_action_prob = tf.Print(log_action_prob, [action_onehot_placeholder], message="onehot placecholder: ",summarize=1000)
        # log_action_prob = tf.Print(log_action_prob, [action_prob_placeholder], message="action placecholder: ",summarize=1000)
        # log_action_prob = tf.Print(log_action_prob, [one_hot_action_probilities], message="pre_sum: ",summarize=1000)
        # log_action_prob = tf.Print(log_action_prob, [action_prob], message="post_sum: ",summarize=1000)
        # log_action_prob = tf.Print(log_action_prob, [log_action_prob], message="log_action_prob: ")
        # log_action_prob = tf.Print(log_action_prob, [discount_reward_placeholder], message="discount_reward_placeholder: ", summarize=1000)

        # The loss is (minimize the loss / maximize the likelihood)
        # log action probability times the discounted reward
        loss = -log_action_prob * discount_reward_placeholder
        # loss = tf.Print(loss, [loss], message="loss: ")
        loss = tf.reduce_mean(loss)
        # loss = tf.Print(log_action_prob, [loss], message="reduced mean: ")

        adam = tf.keras.optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        # Create a function that has inputs, and the update (which is defined by the adam update.)
        # Self model input is another name for the state! Because that is the input
        self.train_fn = tf.keras.backend.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                           outputs=[],
                                           updates=updates)


    def get_action(self, state):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.input_dim,), "{} != {}".format(shape, self.input_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.input_dim), "{} != {}".format(shape, self.input_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.model.predict(state))
        assert len(action_prob) == self.output_dim, "{} != {}".format(len(action_prob), self.output_dim)
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

    def fit(self, states, actions, rewards):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        # Make an action one hot vector
        action_onehot = np_utils.to_categorical(actions, num_classes=self.output_dim)
        discount_reward = compute_discounted_R(rewards)

        self.train_fn([states, action_onehot, discount_reward])


def compute_discounted_R(sampled_rewards, discount_rate=.99):
    """Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    Examples:
        >>> R = [1, 1, 1]
        >>> compute_discounted_R(R, .99) # before normalization
        [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    """
    discounted_r = np.zeros_like(sampled_rewards, dtype=np.float32)
    running_add = 0
    for timestep in reversed(range(len(sampled_rewards))):
        running_add = running_add * discount_rate + sampled_rewards[timestep]
        discounted_r[timestep] = running_add

    # normalization step
    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std()
    return discounted_r

def run_episode(env, agent):
    """Returns an episode reward
    (1) Play until the game is done
    (2) The agent will choose an action according to the policy
    (3) When it's done, it will train from the game play
    Args:
        env (gym.env): Gym environment
        agent (Agent): Game Playing Agent
    Returns:
        total_reward (int): total reward earned during the whole episode
    """
    done = False
    S = []
    A = []
    R = []

    s = env.reset()

    total_reward = 0

    while not done:

        a = agent.get_action(s)

        s2, r, done, info = env.step(a)
        total_reward += r

        S.append(s)
        A.append(a)
        R.append(r)

        s = s2

        if done:
            S = np.array(S)
            A = np.array(A)
            R = np.array(R)

            agent.fit(S, A, R)

    return total_reward

try:
    env = gym.make("CartPole-v0")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = VanillaPolicyAgent(input_dim, output_dim, [16, 16])

    for episode in range(2000):
        reward = run_episode(env, agent)
        print(episode, reward)

finally:
    env.close()
