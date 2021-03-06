import spectrum
import argparse
import os
from spectrum import SpectrumEnv
from spectrum import Suit
from spectrum import Agent
from spectrum import Feature
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import RMSprop
from keras.optimizers import SGD
import sys

"""
Human-designed strategy. Send on two channels, switch channels if it fails.
"""
class TeamPlayer:

    def __init__(self, env=None, num=0, priors=[1,3]):
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
        if my_obs[Feature.INDEX] == self.my_env.sequence_len:
            return [0] * self.my_env.num_channels
        guess = []
        state = self.my_env.action_space.states[my_obs[Feature.STATE]]
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

class HumanSender(TeamPlayer):

    # Send the next card in the sequence on every channel.
    def choose_sender_action(self, my_obs):
        my_seq_code = my_obs[Feature.SEQUENCE]
        my_seq = self.my_env.action_space.seqs[my_seq_code]
        my_seq_index = my_obs[Feature.INDEX]
        action = [my_seq[my_seq_index] for i in range(self.my_env.num_channels)]
        return action

"""
Select actions uniformly at random
"""
class RandomPlayer(TeamPlayer):
    def choose_action(self, observation):
        sample = self.my_env.action_space.sample()
        return sample[self.idnum]

class PsychicNullPlayer(TeamPlayer):
    # def __init__(self, env, num=0, priors=[0]):
    #     TeamPlayer.__init__(self, env, num, priors)
    #
    def choose_action(self, observation):
        if self.my_env.turn == Agent.SENDER:
            return 0
        else:
            seq = self.my_env.sequence_list[self.idnum]
            if len(seq) > self.my_env.num_channels:
                index = self.my_env.indices[self.idnum]
                seq = seq[index:index + self.my_env.num_channels]
            while len(seq) < self.my_env.num_channels:
                seq.append(0)
            return self.my_env.action_space.states.index(tuple(seq))

"""
Neural network player based on https://keon.io/deep-q-learning/
"""
class LearnerPlayer(TeamPlayer):
    def __init__(self, env, idnum, state_size, action_size, savepath="./save/",
                 verbose=1,
                 gamma=0.9, epsilon=1.0,
                 e_decay=0.999,
                 e_min=0.05,
                 learning_rate=0.01, batch_size=64):
        TeamPlayer.__init__(self, env, idnum)
        self.state_size = state_size
        self.action_size = action_size
        self.verbose = verbose
        self.savepath = savepath
        self.memory = deque(maxlen=100000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.e_decay = e_decay
        self.e_min = e_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()

    def set_env(self, env):
        self.my_env = env

    def _build_model(self):
        model = Sequential()
        # Input layer 4 and hidden layer with 20 nodes
        model.add(Dense(128, input_dim=self.my_env.observation_space.shape,
                         activation='sigmoid'))
        # hidden layer with 128 nodes
        model.add(Dense(128, activation='sigmoid'))
        # output layer with 256 nodes
        model.add(Dense(self.my_env.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=SGD(lr=self.learning_rate, decay=1e-6,
                                    momentum=0.9, nesterov=True))
                      # optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, observation):
        my_obs = np.reshape(observation[self.idnum], [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return self.my_env.action_space.sample()[self.idnum]
            # return random.randrange(self.action_size)
        act_values = self.model.predict(my_obs)
        act_index = np.argmax(act_values[0])
        return act_index

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        if batch_size == 0:
            batch_size = 1
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
        self.model.fit(X, Y, batch_size=batch_size, epochs=20, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_opponent(self, alg='metarandom'):
        if alg == 'metarandom':
            alg = random.choice(['alphago','random','hardcode'])
        if alg == 'alphago':
            version = random.choice(os.listdir(self.savepath))
            opponent = LearnerPlayer(self.my_env, 0, self.state_size,
                                     self.action_size)
            opponent.load(self.savepath + version)
            return opponent
        elif alg == 'random':
            return RandomPlayer(self.my_env)
        elif alg == 'hardcode':
            return TeamPlayer(self.my_env)
        elif alg == 'null':
            return PsychicNullPlayer(self.my_env)
        else:
            print("Invalid algorithm {}".format(alg))

    def train(self, episodes=100, maxrounds=100):
        self.my_env = SpectrumEnv()
        self.my_env.set_constants(noise_per_person=0)
        totaltime = 0.0
        for e in range(episodes):
            obs = self.my_env.reset()
            self.player1 = self.get_opponent('null')
            my_obs = obs[self.idnum]
            my_obs = np.reshape(my_obs, [1, self.state_size])
            for time in range(maxrounds):
                action = [self.player1.choose_action(obs),
                          self.choose_action(obs)]
                obs, reward, done, _ = self.my_env.step(action)
                # reward = 0
                reward = reward / (time + 1)
                my_next_obs = np.reshape(obs[self.idnum],
                                         [1, self.state_size])
                if random.random() < 0.05:
                    self.remember(my_obs, action, reward, my_next_obs, done)
                my_obs = my_next_obs
                if done or time == maxrounds - 1:
                    if time % 10 == 0:
                        if self.verbose >= 1:
                            print("Episode: {}/{}, score: {}, reward: {}, e: {:.2}"
                                  .format(e, episodes, self.my_env.roundnum, reward, self.epsilon))
                    totaltime += self.my_env.roundnum
                    break
            self.replay(self.batch_size)
            if e % 10 == 0:
                if self.verbose >= 1:
                    avg = totaltime / (e + 1)
                    print(avg)
                #self.save(args.savepath+"spectrum.{}.h5".format(e))
        self.my_env.close()

class LearnerReceiverPlayer(LearnerPlayer):

    def train(self, episodes=100, maxrounds=100):
        self.my_env = SpectrumEnv()
        self.my_env.set_constants(num_pairs=1, noise_per_person=0, \
                                  num_channels=4, sequence_len=5, \
                                  noise_change_prob=0)
        self.my_env.reset()
        sender_player = HumanSender(self.my_env, 0)
        cum_sum_rounds = 0

        for e in range(episodes):
            sen_obv, sen_obv_nn, rec_obv, rec_obv_nn = None, None, None, None
            sen_obv = self.my_env.reset() # Initial sender state
            for t in range(maxrounds):
                # Send
                sen_action = [sender_player.choose_action(sen_obv)]
                rec_obv, reward, done, _ = self.my_env.step(sen_action)
                rec_obv_nn = np.reshape(rec_obv, [1, self.state_size])
                # Receive
                rec_action = [self.choose_action(rec_obv_nn)]
                sen_obv, reward, done, _ = self.my_env.step(rec_action)
                sen_obv_nn = np.reshape(sen_obv, [1, self.state_size])
                if reward == 0:
                    reward = -t
                else:
                    reward = 5
                #reward = reward / (t + 1)**4
                if random.random() < 1.0:
                    self.remember(rec_obv_nn, rec_action, reward, sen_obv_nn, done)
                if (done or t == maxrounds - 1):
                    #if self.verbose >= 1 and e % 10 == 0:
                    #    print("Episode: {}/{}, score: {}, reward: {}, e: {:.2}"
                    #          .format(e, episodes, self.my_env.roundnum, reward, #self.epsilon))
                    #    sys.stdout.flush()
                    cum_sum_rounds += self.my_env.roundnum
                    break
            if e % 10 == 0:
                self.replay(self.batch_size)
                if e % 100 == 0:
                    avg = cum_sum_rounds / (e + 1)
                    print("Episode {}, avg score {:.4}, e {:.4}".format(e, avg, self.epsilon))
                    sys.stdout.flush()
        self.my_env.close()


def fake_train(env, player1, player2, episodes, maxrounds, verbose=1):
    totaltime = 0.0
    obs = env.reset()
    print( len(env.action_space.states))
    for e in range(episodes):
        obs = env.reset()
        for time in range(maxrounds):
            action = [player1.choose_action(obs),
                      player2.choose_action(obs)]
            obs, reward, done, _ = env.step(action)
            reward = reward / (time + 1)
            if done or time == maxrounds - 1:
                # if time % 100 == 0:
                if verbose >= 1:
                    print("Episode: {}/{}, score: {}, reward: {}"
                          .format(e, episodes, env.roundnum, reward))
                totaltime += env.roundnum
                break
    avg = totaltime / episodes
    print ("average: {}".format(avg))

def main(args):
    env = SpectrumEnv()
    env.set_constants(num_pairs=1, noise_per_person=0, \
                              num_channels=4, sequence_len=5, \
                              noise_change_prob=0)
    env.reset()
    state_size = env.observation_space.shape
    receiver_player = LearnerReceiverPlayer(env, 0, state_size, \
                                            env.action_size, args.savepath, \
                                            args.verbose)
    receiver_player.train(1000000, 10000)

"""
def main(args):
    env = SpectrumEnv()
    env.set_constants(num_channels=4, noise_per_person=0, sequence_len=5)
    # player1 = HumanReceiver(env)
    observation = env.reset()
    player1 = PsychicNullPlayer(env)
    # player2 = PsychicNullPlayer(env,1)
    # player2 = RandomPlayer(env, 1)
    state_size = env.observation_space.shape
    player2 = LearnerPlayer(env, 1, state_size, env.action_size, args.savepath,
                            args.verbose)
    if args.nn != '':
        player2.load(args.nn)
    else:
        player2.train(10000, 10000)
    player2.set_env(env)
    # player2 = TeamPlayer(env, 1, [0,2])
    # fake_train(env, player1, player2, 100, 10000, args.verbose)
    observation = env.reset()
    # for t in range(10):
    #     action1 = player1.choose_action(observation)
    #     action2 = player2.choose_action(observation)
    #     action = [action1, action2]
    #     if args.verbose >= 2:
    #         print "round: {}".format(env.roundnum)
    #         env.render('god')
    #         print (env.format_obs(observation))
    #         print (env.format_action(action))
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         if args.verbose >= 1:
    #             print (("Episode finished after {} timesteps".format(t+1)))
    #             print (env.format_obs(observation))
    #             action1 = player1.choose_action(observation)
    #             print (env.format_action([action1]))
    #             break
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", metavar="Verbosity", type=int, default=0,
                        help="0 for silent, 1 for per episode,"+
                        "2 for god mode per round, 3 for internals")
    parser.add_argument("--nn", metavar="NeuralNetFile", type=str, default='',
                        help="path to the .h5 file containing neural net data")
    parser.add_argument("--savepath", metavar="directory", type=str,
                        default='./save/',
                        help="directory where .h5 neural nets are stored")
    args = parser.parse_args()
    main(args)
