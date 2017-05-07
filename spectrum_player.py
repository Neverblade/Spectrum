import spectrum
from spectrum import SpectrumEnv
from spectrum import Suit
from spectrum import Agent

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
            return self.choose_sender_action(my_obs)
        elif self.my_env.turn == Agent.RECEIVER:
            return self.choose_receiver_action(my_obs)
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
                if i == my_obs["STATE"][i] and \
                    my_obs["STATE"][i] != Suit.NULL:
                    # our last message went through,
                    # keep using this channel.
                    channels += i
            if channels == []:
                # No channels; add any blank channels
                for i in range(len(my_obs["STATE"])):
                    if my_obs["STATE"][i] == Suit.NULL:
                        channels += i
            if channels == []:
                # Still no channels; get aggressive
                channels == [0,1,2,3]
        if len(my_obs["SEQUENCE"]) > my_obs["INDEX"]:
            for i in channels:
                # We're done, try to help the other team?
                action[i] = my_obs["SEQUENCE"][my_obs["INDEX"]]
        print("Sender "+str(self.idnum)+" sending "+str(action))
        return action

    def choose_receiver_action(self, my_obs):
        # Not allowed to use self.last_action
        # TODO: This currently doesn't adapt to the variations in the sender
        # version
        guess = []
        for i in self.priors:
            if my_obs["STATE"][i] != Suit.NULL:
                guess.append(my_obs["STATE"][i])
        return guess

class HumanReceiver(TeamPlayer):
    def choose_receiver_action(self, my_obs):
        # Prompt reader for a single guess.
        # TODO: input is not validated, and multiple guesses are not supported.
        self.my_env.render('human')
        guess = input("Guess? ")
        return [guess]


"""
Select actions uniformly at random
"""
class RandomPlayer(TeamPlayer):
    def choose_action(self, observation):
        return(self.my_env.action_space.sample()[self.idnum])


def main():
    print("Running")
    env = SpectrumEnv()
    # player1 = TeamPlayer(env)
    player1 = HumanReceiver(env)
    player2 = RandomPlayer(env, 1)
    # player2 = TeamPlayer(env, 1, [0,2])
    observation = env.reset()
    for t in range(100):
        print("Observation:",observation)
        action1 = player1.choose_action(observation)
        action2 = player2.choose_action(observation)
        action = [action1, action2]
        print("Action:", [action1, action2])
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode finished after {} timesteps".format(t+1))
            break

if __name__ == '__main__':
    main()
