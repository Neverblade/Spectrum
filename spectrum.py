from gym import *
from gym.utils import seeding
import numpy as np
import random

"""

=== DOCUMENTATION FOR SPECTRUM ===

The process runs as follows:

1. Initialize Spectrum object
2. Call reset, receive initial list of Sender OBV
Game Starts
    1. Senders sees their OBVs, decides actions, submits to step()
    2. step() returns list of RECEIVER OBVs, reward of 0
    3. Receivers sees their OBVs, decides actions, submits to step()
    4. step() returns list of SENDER OBVs, reward of x
    5. Back to (1)

What is a Sender OBV? A dictionary of the following items:
    SEQUENCE: The sequence of suits the sender needs to transmit.
        Object: List of length (sequence_len) of elements from Suit.SUITS.
    INDEX: What position in the sequence the pair is on.
        Object: Integer.
    ROUND: The current round of the game.
        Object: Integer.
    NOISE: The noise experienced by each pair.
        Object: boolean 2D list of size (num_pairs x num_channels).
        Each row represents a pair, columns represent the channel. If true there's noise.
    PREV_STATE: The previous round's board state (after submissions, conflict, and noise processing).
        Object: List of length (num_channels) of elements from Suit.ALL.

What is a Sender ACTION?
    A choice for each channel of what, if any, suit to send.
    Object: List of length (num_channels) of elements from Suit.ALL

What is a Receiver OBV? A dictionary of the following items:
    INDEX: What position in the sequence the pair is on.
        Object: Integer.
    ROUND: The current round of the game.
        Object: Integer.
    STATE: The board state (after submissions, conflict, and noise processing).
        Object: List of length (num_channels) of elements from Suit.ALL.

What is a Receiver ACTION?
    A guess of what suits the sender was trying to convey.
    Object: List of any length of elements from Suit.SUITS

What is the internal state of the game? The following attributes:
    TURN: Which agent's turn it is to act.
        Object: Either Agent.SENDER or Agent.RECEIVER.
    ROUND: What round in the game it is.
        Object: Integer.
    SEQUENCE_LIST: The sequences of suits each sender needs to transmit.
        Object: 2D list of size (num_pairs x sequence_len), where each element is from
        Suit.SUITS. Each row represents a pair.
    NOISE_LIST: The noise experienced by each pair.
        Object: Boolean 2D list of size (num_pairs x num_channels).
        Each row represents a pair, columns represent the channel. If true there's noise.
    INDICES: The position in their sequence each pair is on.
        Object: Int list of length (num_pairs).
    PREV_STATE: The previous board state (post-processed) from the Senders.
        Object: Object: List of length (num_channels) of elements from Suit.ALL.

Misc Things
    1. What gets sent and returned from step() are always LISTS of OBVs and actions.
"""
class Spectrum(Env):

    def __init__(self):
        # Setting arbitrary constants
        self.num_pairs = 2
        self.noise_per_person = 1
        self.num_channels = num_pairs * 2
        self.sequence_len = 5

        # Setting up internal state.
        self.turn = None
        self.round = None
        self.sequence_list = [None for i in range(num_pairs)]
        self.noise_list = [None for i in range(num_pairs)]
        self.indices = [None for i in range(num_pairs)]
        self.prev_state = None

        # Seed randomly.
        self._seed()


    def _reset(self):
        # Assign attributes accordingly
        self.turn = Agent.SENDER
        self.round = 1
        for i in range(num_pairs):
            sequence_list[i] = [Suit.sample_suits() for j in range(sequence_len)]
            noise_list[i] = [True if j < noise_per_person else False for j in range(num_channels)]
            random.shuffle(noise_list[i])
            indices[i] = 0
        self.prev_state = [Suit.NULL for i in range(num_channels)]

        # Return initial sender OBVs
        obv_list = []
        for i in range(num_pairs):
            obv = {}
            obv["SEQUENCE"] = self.sequence_list[i]
            obv["INDEX"] = self.indices[i]
            obv["ROUND"] = self.round
            obv["NOISE"] = self.noise_list[i]
            obv["PREV_STATE"] = self.prev_state
            obv_list.append(obv)
        return obv_list


    def _seed(self, seed=None):
        random.seed(seed)


    def _step(self, action_list):
        if self.turn == Agent.SENDER:
            obv_list = [] # Receiver OBV list
            board_state = [Suit.NULL for i in range(num_channels)]
            for i in range(num_pairs):
                for j in range(num_channels):
                    if self.noise_list[i][j]:
                        continue
                    suit = action_list[i][j]
                    if suit != Suit.NULL:
                        if board_state[j] == Suit.NULL: # Free channel
                            board_state[j] = suit
                        else: # Collision!
                            board_state[j] = Suit.NULL
                obv = {"STATE": board_state, "INDEX": self.indices[i], "ROUND": self.round}
                obv_list.append(obv)
            self.prev_state = board_state
            return obv_list, 0, False, {}
        else:
            obv_list, reward, done = [], 0, True
            for i in range(num_pairs):
                # Process action
                seq_guess, index = action_list[i], self.indices[i]
                for j in range(len(seq_guess)):
                    if index + j >= sequence_len \
                       or seq_guess[j] != self.sequence_list[i][index + j]:
                        break
                    else:
                        self.indices[i] += 1
                        reward += 1

                # Check if done
                if self.indices[i] != sequence_len:
                    done = False

                # Construct OBV
                obv = {}
                obv["SEQUENCE"] = self.sequence_list[i]
                obv["INDEX"] = self.indices[i]
                obv["ROUND"] = self.round
                obv["NOISE"] = self.noise_list[i]
                obv["PREV_STATE"] = self.prev_state
                obv_list.append(obv)
            return obv_list, reward, done, {}


    def _render(self):
        pass


"""
Actions for both senders and receivers are described above.
"""
class ActionSpace(Space):

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.pair_index = pair_index

    def sample(self):
        if spectrum.turn == Agent.SENDER:
            actions = []
            for i in range(spectrum.num_pairs):
                actions.append([Suit.sample_all() for j in range(spectrum.num_channels)])
        else:
            actions = []
            for i in range(spectrum.num_pairs):
                num_cards_left = spectrum.sequence_len - spectrum.indices[i]
                len = random.randint(1, num_cards_left + 1)
                actions.append([Suit.sample_suits() for j in range(len)])
        return actions

    def contains(self, x):
        if len(x) != spectrum.num_pairs:
            return False
        if spectrum.turn == Agent.SENDER:
            for j in range(spectrum.num_pairs):
                if len(x[j]) != spectrum.num_channels:
                    return False
                for i in range(spectrum.num_channels):
                    if x[j][i] not in Suit.ALL:
                        return False
            return True
        else:
            for j in range(spectrum.num_pairs):
                for i in range(len(x[j])):
                    if x[j][i] not in Suit.SUITS:
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
        return random.choice(SUITS)

    @staticmethod
    def sample_all():
        return random.choice(ALL)


"""
Enum replacement for agents in the game.
"""
class Agent:
    SENDER = 0
    RECEIVER = 1