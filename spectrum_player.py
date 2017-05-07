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


def main():
    env = SpectrumEnv()
    # player1 = TeamPlayer(env)
    player1 = HumanReceiver(env)
    player2 = RandomPlayer(env, 1)
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
