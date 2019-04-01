import numpy as np
from keras.layers import Activation, Dense, Input, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


class A2CAgent:
    def __init__(self, game_length, action_size, n_options):
        self.render = True
        self.load_model = False
        self.game_length = game_length
        self.action_size = action_size
        self.value_size = 1
        self.n_options = n_options

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/Cowbull_actor.h5")
            self.critic.load_weights("./save_model/Cowbull_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        state = Input(
            shape=((
                self.game_length * (1 + self.action_size) * self.n_options),))
        x = Dense(
            128, activation='relu', kernel_initializer='he_uniform')(state)
        # x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        # x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)

        x1 = Dense(
            self.n_options,
            activation='softmax',
            kernel_initializer='he_uniform')(x)
        x2 = Dense(
            self.n_options,
            activation='softmax',
            kernel_initializer='he_uniform')(x)
        x3 = Dense(
            self.n_options,
            activation='softmax',
            kernel_initializer='he_uniform')(x)
        x4 = Dense(
            self.n_options,
            activation='softmax',
            kernel_initializer='he_uniform')(x)

        output = concatenate([x1, x2, x3, x4])

        actor = Model(inputs=state, outputs=output)

        actor.summary()
        actor.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        state = Input(
            shape=((
                self.game_length * (1 + self.action_size) * self.n_options),))
        x = Dense(
            128, activation='relu', kernel_initializer='he_uniform')(state)
        # x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        # x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        output = Dense(
            self.value_size,
            activation='linear',
            kernel_initializer='he_uniform')(x)

        critic = Model(inputs=state, outputs=output)

        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))

        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):

        action = []
        l1 = to_categorical(
            state[:, :self.action_size], num_classes=self.n_options)
        l2 = np.zeros((self.game_length, 1, self.n_options))
        l2[:, 0, 0] = state[:, self.action_size]
        state = np.concatenate((l1, l2), axis=1)
        state = np.reshape(
            state,
            (1, (self.game_length * (1 + self.action_size) * self.n_options)))

        policy = self.actor.predict(state, batch_size=1).flatten()
        for i in range(self.action_size):

            action.append(
                np.random.choice(
                    np.arange(0, self.n_options),
                    p=policy[i * self.n_options:(i + 1) * self.n_options]))

        return action

    # update policy network every episode
    def train_model(self, state, action, reward, next_state):

        l1 = to_categorical(
            state[:, :self.action_size], num_classes=self.n_options)
        l2 = np.zeros((self.game_length, 1, self.n_options))
        l2[:, 0, 0] = state[:, self.action_size]
        state = np.concatenate((l1, l2), axis=1)

        l1 = to_categorical(
            next_state[:, :self.action_size], num_classes=self.n_options)
        l2 = np.zeros((self.game_length, 1, self.n_options))
        l2[:, 0, 0] = next_state[:, self.action_size]
        next_state = np.concatenate((l1, l2), axis=1)

        state = np.reshape(
            state,
            (1, (self.game_length * (1 + self.action_size) * self.n_options)))
        next_state = np.reshape(
            next_state,
            (1, (self.game_length * (1 + self.action_size) * self.n_options)))

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((self.action_size, self.n_options))

        value = self.critic.predict(state)[0]

        next_value = self.critic.predict(next_state)[0]

        for i in range(self.action_size):
            advantages[i][action[i]] = reward + self.discount_factor * (
                next_value) - value

        advantages = np.reshape(advantages,
                                (1, (self.action_size * self.n_options)))

        target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)