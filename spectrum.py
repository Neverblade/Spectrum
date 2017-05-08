from gym import *
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import itertools

"""

=== DOCUMENTATION FOR SPECTRUM ===

The process runs as follows:

1. Initialize SpectrumEnv object
2. Call set_constants() to shape game as desired.
3. Call reset, receive initial list of Sender OBV
Game Starts
    1. Senders sees their OBVs, decides actions, submits to step()
    2. step() returns list of RECEIVER OBVs, reward of 0
    3. Receivers sees their OBVs, decides actions, submits to step()
    4. step() returns list of SENDER OBVs, reward of x
    5. Back to (1)

What is a Sender OBV? A list of the following items:
    SEQUENCE: The sequence of suits the sender needs to transmit.
        Object: An int corresponding to a position in a one-hot
        encoding list of all possible suit sequences.
    INDEX: What position in the sequence the pair is on.
        Object: Integer.
    NOISE: The noise experienced the sender.
        Object: An int corresponding to a position in a one-hot
        encoding list of all possible noise combinations.
    PREV_STATE: The previous round's board state (after submissions, conflict,
        and noise processing).
        Object: An int corresponding to a position in a one-hot
        encoding list of all possible states.

What is a Sender ACTION?
    A choice for each channel of what, if any, suit to send.
    Underlying structure: List of length (num_channels) of elements from Suit.ALL
    Object: An int corresponding to its position in a one-hot encoding list
    of all possible states.

What is a Receiver OBV? A dictionary of the following items:
    INDEX: What position in the sequence the pair is on.
        Object: Integer.
    STATE: The board state (after submissions, conflict, and noise processing).
        Object: An int corresponding to a position in a one-hot
        encoding list of all possible states.

What is a Receiver ACTION?
    A guess of what suits the sender was trying to convey.
    Underlying Structure: List of of elements from Suit.SUITS
    Object: AN int corresponding to its positon in a one-hot encoding list
    of all possible sequences.

What is the internal state of the game? The following attributes:
    TURN: Which agent's turn it is to act.
        Object: Either Agent.SENDER or Agent.RECEIVER.
    ROUND: What round in the game it is.
        Object: Integer.
    SEQUENCE_LIST: The sequences of suits each sender needs to transmit.
        Object: 2D list of size (num_pairs x sequence_len), where each element
        is from Suit.SUITS. Each row represents a pair.
    NOISE_LIST: The noise experienced by each pair.
        Object: Boolean 2D list of size (num_pairs x num_channels).
        Each row represents a pair, columns represent the channel. If true
        there's noise.
    INDICES: The position in their sequence each pair is on.
        Object: Int list of length (num_pairs).
    PREV_STATE: The previous board state (post-processed) from the Senders.
        Object: Object: List of length (num_channels) of elements from Suit.ALL.

Misc Things
    1. What gets sent and returned from step() are always LISTS of OBVs and
       actions.
"""

class SpectrumEnv(Env):

    metadata = {'render.modes': ['ansi', 'human', 'god']}

    """
    Initialization doesn't directly allow for playing. Need to call reset.
    """
    def __init__(self):
        # Setting arbitrary constants
        self.num_pairs = 2
        self.noise_per_person = 1
        self.num_channels = self.num_pairs * 2
        self.sequence_len = 5
        self.noise_change_prob = 0

    """
    Set new values for the constants.
    """
    def set_constants(self, num_pairs=2, noise_per_person=1, num_channels=4,
                      sequence_len=5, noise_change_prob=0):
        self.num_pairs = num_pairs
        self.noise_per_person = noise_per_person
        self.num_channels = num_channels
        self.sequence_len = sequence_len
        self.noise_change_prob = noise_change_prob

    """
    Sets up the game for playing.
    """
    def _reset(self):
        # Setting up internal state.
        self.turn = None
        self.roundnum = None
        self.sequence_list = [None for i in range(self.num_pairs)]
        self.noise_list = [None for i in range(self.num_pairs)]
        self.indices = [0 for i in range(self.num_pairs)]
        self.prev_state = None
        self.guessed = [[] for i in range(self.num_pairs)]

        # Set up spaces/shapes
        self.noises = [i for i in itertools.product(range(2),
                                               repeat=self.num_channels)]

        # Set up action space
        self.action_space = ActionSpace(self)
        self.action_size = len(self.action_space.states)

        # Misc stuff.
        self._seed()
        self.spec = None
        self.observation_space = spaces.MultiDiscrete(
            [[0, len(self.action_space.seqs)],    # Sequence
             [0, len(self.action_space.states)],  # State
             [0, self.sequence_len],    # Index
             [0, len(self.noises)]]) # Noise

        # Assign attributes accordingly
        self.turn = Agent.SENDER
        self.roundnum = 1
        for i in range(self.num_pairs):
            self.sequence_list[i] = \
                [Suit.sample_suits() for j in range(self.sequence_len)]
            self.noise_list[i] = [True if j < self.noise_per_person
                                  else False for j in range(self.num_channels)]
            random.shuffle(self.noise_list[i])
            self.indices[i] = 0
        self.prev_state = [Suit.NULL for i in range(self.num_channels)]

        # Return initial sender OBVs
        obv_list = []
        for i in range(self.num_pairs):
            obv = [0] * self.observation_space.shape
            obv[Feature.SEQUENCE] = \
                self.action_space.seqs.index(tuple(self.sequence_list[i]))
            my_state = tuple(self.prev_state)
            obv[Feature.STATE] = self.action_space.states.index(my_state)
            obv[Feature.INDEX] = self.indices[i]
            obv[Feature.NOISE] = self.noises.index(tuple(self.noise_list[i]))
            obv = np.array(obv)
            obv_list.append(obv)
        return obv_list


    def _seed(self, seed=None):
        random.seed(seed)

    """
    See the procedure (A) for how to use step() properly.
    Input a list of actions (either Sender Actions (B) or Receiver Actions (D)).
    Takes the corresponding action and returns either a list of Sender OBVs or
    Receiver OBVs.
    """
    def _step(self, action_list):
        assert self.action_space.contains(action_list), \
            "%r (%s) invalid"%(action_list, type(action_list))
        if self.turn == Agent.SENDER:
            obv_list = [] # Receiver OBV list
            board_state = [Suit.NULL for i in range(self.num_channels)]
            for i in range(self.num_pairs):
                action = self.action_space.states[action_list[i]]
                for j in range(self.num_channels):
                    noise = self.noise_list[i][j]
                    if noise:
                        continue
                    suit = action[j]
                    if suit != Suit.NULL:
                        if board_state[j] == Suit.NULL: # Free channel
                            board_state[j] = suit
                        else: # Collision!
                            board_state[j] = Suit.NULL
                obv = [0] * self.observation_space.shape
                obv[Feature.SEQUENCE] = 0
                obv[Feature.INDEX] = self.indices[i]
                obv[Feature.STATE] = \
                    self.action_space.states.index(tuple(board_state))
                obv[Feature.NOISE] = 0
                obv = np.array(obv)
                obv_list.append(obv)
            self.prev_state = board_state
            self.turn = Agent.RECEIVER
            return obv_list, 0, False, {}
        else:
            obv_list, reward, done = [], 0.0, True
            for i in range(self.num_pairs):
                # Process action
                seq, index = action_list[i], self.indices[i]
                seq_guess = list(filter(lambda a: a != 0,
                                        self.action_space.seqs[seq]))
                for j in range(len(seq_guess)):
                    if index + j >= self.sequence_len \
                       or seq_guess[j] != self.sequence_list[i][index + j]:
                        reward -= 1.0 / len(Suit.SUITS)
                        break
                    else:
                        self.guessed[i].append(seq_guess[j])
                        self.indices[i] += 1
                        reward += 1

                # Check if done
                if self.indices[i] != self.sequence_len:
                    done = False
                self.roundnum += 1

                # Construct OBV
                obv = [0] * self.observation_space.shape
                obv[Feature.SEQUENCE] = \
                    self.action_space.seqs.index(tuple(self.sequence_list[i]))
                obv[Feature.STATE] = self.action_space.states.index(tuple(self.prev_state))
                obv[Feature.INDEX] = self.indices[i]
                obv[Feature.NOISE] = \
                                    self.noises.index(tuple(self.noise_list[i]))
                obv = np.array(obv)
                obv_list.append(obv)

                # Update noises
                for i in range(self.num_pairs):
                    if random.random() < self.noise_change_prob:
                        self.noise_list[i] = [True if j < self.noise_per_person
                                  else False for j in range(self.num_channels)]

            self.turn = Agent.SENDER
            return obv_list, reward, done, {}


    def _render(self, mode='god', close=False):
        s = "\n"
        s += "=== Round: " + str(self.roundnum)
        s += " Turn: " + ("Sender" if self.turn == Agent.SENDER else "Receiver") + " ===\n\n"

        s += "Previous State:\n"
        for i in range(self.num_channels):
            s += "    " + str(self.prev_state[i])
        s += "\n\n"

        if mode == 'god':
            s += "Sequences:\n"
            for i in range(self.num_pairs):
                s += "    Sender " + str(i) + ": "
                for j in range(self.sequence_len):
                    if self.indices[i] == j:
                        s += ">" + str(self.sequence_list[i][j]) + "< "
                    else:
                        s += str(self.sequence_list[i][j]) + " "
                if self.indices[i] == self.sequence_len:
                    s += "DONE!"
                s += "\n\n"

            s += "Noise:\n"
            for i in range(self.num_pairs):
                for j in range(self.num_channels):
                    s += "    " + ("X" if self.noise_list[i][j] else ".")
                s += "\n\n"
        else:
            s += "Global score is {}\n".format(sum(self.indices))


        s += "============================="
        if mode == 'human':
            print(s)
        else:
            return s


"""
Actions for both senders and receivers are described above.
"""
class ActionSpace(Space):

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.states = [i for i in itertools.product(range(len(Suit.ALL)),
                                   repeat=self.spectrum.num_channels)]
        self.seqs = [i for i in itertools.product(range(1+len(Suit.SUITS)),
                                 repeat=self.spectrum.sequence_len)]

    def sample(self):
        if self.spectrum.turn == Agent.SENDER:
            actions = []
            for i in range(self.spectrum.num_pairs):
                actions.append(
                    self.states.index(tuple([Suit.sample_all() for j in
                                range(self.spectrum.num_channels)])))
        else:
            actions = []
            for i in range(self.spectrum.num_pairs):
                num_cards_left = self.spectrum.sequence_len - \
                                    self.spectrum.indices[i]
                length = random.randint(0, self.spectrum.num_channels)
                guess = [Suit.sample_suits() for j in range(length)]
                while len(guess) < self.spectrum.num_channels:
                    guess.append(0)
                actions.append(self.states.index(tuple(guess)))
        return actions

    def contains(self, x):
        #try:
        if self.spectrum.turn == Agent.SENDER:
            lst = range(len(self.states))
        else:
            lst = range(len(self.seqs))
        for i in range(len(x)):
            if x[i] not in lst:
                return False
        return True

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

"""
Enum replacement for possible channel containers
"""
class Suit:
    NULL = 0
    DIAMOND = 1
    HEART = 2
    CLUB = 3
    SPADE = 4

    SUITS = [DIAMOND, HEART, CLUB, SPADE]
    ALL = SUITS + [NULL]

    @staticmethod
    def sample_suits():
        return random.choice(Suit.SUITS)

    @staticmethod
    def sample_all():
        return random.choice(Suit.ALL)


"""
Enum replacement for agents in the game.
"""
class Agent:
    SENDER = 0
    RECEIVER = 1

class Feature:
    INDEX = 0
    STATE = 1
    NOISE = 2
    SEQUENCE = 3
