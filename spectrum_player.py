import spectrum
from spectrum import SpectrumEnv
from spectrum import Suit
from spectrum import Agent
from spectrum import Feature
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

EPISODES = 100

"""
Human-designed strategy. Send on two channels, switch channels if it fails.
"""
class TeamPlayer:

    def __init__(self, env, num=0, priors=[1,3]):
        self.my_env = env
        self.idnum = num
        self.last_action = None
        self.priors = priors

    def choose_action(self, observation):
        my_obs = observation[self.idnum]
        if self.my_env.turn == Agent.SENDER:
            return self.my_env.action_space.states.index(
                tuple(self.choose_sender_action(my_obs)))
        elif self.my_env.turn == Agent.RECEIVER:
            return self.my_env.action_space.states.index(
                tuple(self.choose_receiver_action(my_obs)))
        else:
            print("Error: invalid turn")
            return

    def choose_sender_action(self, my_obs):
        action = [0 for _ in range(self.my_env.num_channels)]
        channels = []
        if self.last_action == None:
            channels = self.priors
        else:
            for i in self.last_action:
                if i == my_obs[Feature.STATE][i] and \
                    my_obs[Feature.STATE][i] != Suit.NULL:
                    # our last message went through,
                    # keep using this channel.
                    channels += i
            if channels == []:
                # No channels; add any blank channels
                for i in range(len(my_obs[Feature.STATE])):
                    if my_obs[Feature.STATE][i] == Suit.NULL:
                        channels += i
            if channels == []:
                # Still no channels; get aggressive
                channels == [0,1,2,3]
        my_seq_code = my_obs[Feature.SEQUENCE]
        my_seq = self.my_env.action_space.seqs[my_seq_code]
        my_seq_index = my_obs[Feature.INDEX]
        if my_seq_index == len(my_seq):
            return [0,0,0,0]
        for i in channels:
            # We're done. TODO: try to help the other team?
            next_in_seq = my_seq[my_seq_index]
            action[i] = next_in_seq
        return action

    def choose_receiver_action(self, my_obs):
        # Not allowed to use self.last_action
        # TODO: This currently doesn't adapt to the variations in the sender
        # version
        guess = []
        state = \
            self.my_env.action_space.states[my_obs[Feature.STATE]]
        for i in self.priors:
            if state[i] != Suit.NULL:
                guess.append(state[i])
        while len(guess) < self.my_env.num_channels:
            guess.append(0)
        return guess

class HumanReceiver(TeamPlayer):
    def choose_receiver_action(self, my_obs):
        # Prompt reader for a single guess.
        # TODO: input is not validated, and multiple guesses are not supported.
        self.my_env.render('human')
        guess = []
        guess.append(input("Guess? "))
        while len(guess) < self.my_env.num_channels:
            guess.append(0)
        return guess

"""
Select actions uniformly at random
"""
class RandomPlayer(TeamPlayer):
    def choose_action(self, observation):
        sample = self.my_env.action_space.sample()
        return sample[self.idnum]

"""
Neural network player based on https://keon.io/deep-q-learning/
"""
class LearnerPlayer(TeamPlayer):
    def __init__(self, env, idnum, state_size, action_size,
                 gamma=0.9, epsilon=1.0, e_decay=0.99, e_min=0.05,
                 learning_rate=0.01):
        TeamPlayer.__init__(self, env, idnum)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.e_decay = e_decay
        self.e_min = e_min
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def set_env(self, env):
        self.my_env = env

    def _build_model(self):
        model = Sequential()
        # Input layer 4 and hidden layer with 20 nodes
        model.add(Dense(20, input_dim=4, activation='tanh'))
        # hidden layer with 128 nodes
        model.add(Dense(20, activation='tanh'))
        # output layer with 256 nodes
        model.add(Dense(625, activation='softmax'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, observation):
        my_obs = np.reshape(observation[self.idnum], [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(my_obs)
        act_index = np.argmax(act_values[0])
        return act_index

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self):
        self.my_env = SpectrumEnv()
        self.player1 = RandomPlayer(self.my_env)
        for e in range(EPISODES):
            obs = self.my_env.reset()
            my_obs = obs[self.idnum]
            my_obs = np.reshape(my_obs, [1, self.state_size])
            for time in range(1000):
                # self.my_env.render('human')
                action = [self.player1.choose_action(obs),
                          self.choose_action(obs)]
                next_obs, reward, done, _ = self.my_env.step(action)
                reward = reward if not done else 5
                my_next_obs = np.reshape(next_obs[self.idnum],
                                         [1, self.state_size])
                self.remember(my_obs, action, reward, my_next_obs, done)
                my_obs = my_next_obs
                if done or time == 999:
                    print("Episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, time, self.epsilon))
                    break
            self.replay(32)
            if e % 10 == 0:
                self.save("./save/spectrum.h5")


def main():
    env = SpectrumEnv()
    # player1 = TeamPlayer(env)
    player1 = HumanReceiver(env)
    state_size = env.observation_space.shape
    player2 = LearnerPlayer(env, 1, state_size, env.action_size)
    # player2.load("./save/spectrum.h5")
    player2.train()
    player2.set_env(env)
    # player2 = TeamPlayer(env, 1, [0,2])
    observation = env.reset()
    for t in range(100):
        # env.render('human')
        action1 = player1.choose_action(observation)
        action2 = player2.choose_action(observation)
        action = [action1, action2]
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode finished after {} timesteps".format(t+1))
            break

if __name__ == '__main__':
    main()
