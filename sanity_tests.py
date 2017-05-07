import unittest
import spectrum
from spectrum import Spectrum
from spectrum import Suit

class SpectrumTestMethods(unittest.TestCase):

    """
    Test that a sender can put a card down, and then a receiver can successfuly guess it.
    """
    def test_basic_1(self):
        game, sender_obv = self.setup(1, 0, 4, 1)
        card = sender_obv[0]["SEQUENCE"][0]

        # Perform sender action and check state
        sender_action = [[Suit.NULL, card, Suit.NULL, Suit.NULL]]
        obv, reward, done, info = game.step(sender_action)
        state = obv[0]["STATE"]
        self.assertListEqual(game.prev_state, sender_action[0])
        self.assertListEqual(state, sender_action[0])
        self.assertEqual(reward, 0)
        self.assertEqual(done, False)

        # Receiver guesses
        receiver_action = [[card]]
        obv, reward, done, info = game.step(receiver_action)
        self.assertEqual(obv[0]["INDEX"], 1)
        self.assertEqual(reward, 1)
        self.assertEqual(done, True)

    """
    Test that noise blocks cards.
    """
    def test_noise(self):
        game, sender_obv = self.setup(1, 4, 4, 1) # num_noise = num_channels
        card = sender_obv[0]["SEQUENCE"][0]

        # Sender attempts to send on everything, check state
        sender_action = [[card, card, card, card]]
        obv, reward, done, info = game.step(sender_action)
        state = obv[0]["STATE"]
        self.assertListEqual(game.prev_state, [Suit.NULL for i in range(4)])

    """
    Test that two senders can send stuff w/ and w/o conflict.
    Test that receivers can guess right and wrong and get the correct group score.
    """
    def test_pair_interactions(self):
        game, sender_obv = self.setup(2, 0, 4, 1) # num_noise = num_channels
        card0, card1 = sender_obv[0]["SEQUENCE"][0], sender_obv[1]["SEQUENCE"][0]

        # Senders send on a conflicting channel and a non-conflicting one.
        sender_action = [[Suit.NULL, card0, card0, Suit.NULL], [Suit.NULL, card1, Suit.NULL, card1]]
        obv, reward, done, info = game.step(sender_action)
        self.assertListEqual(obv[0]["STATE"], obv[1]["STATE"])
        self.assertListEqual(obv[0]["STATE"], game.prev_state)
        self.assertListEqual(obv[0]["STATE"], [Suit.NULL, Suit.NULL, card0, card1])

        # One receiver guesses right, the other guesses wrong.
        receiver_action = [[card0], [card1 % 4 + 1]]
        obv, reward, done, info = game.step(receiver_action)
        self.assertEqual(obv[0]["INDEX"], 1)
        self.assertEqual(obv[1]["INDEX"], 0)
        self.assertEqual(reward, 1)
        self.assertEqual(done, False)

    def setup(self, num_pairs=2, noise_per_person=1, num_channels=4, sequence_len=5):
        game = Spectrum()
        game.set_constants(num_pairs, noise_per_person, num_channels, sequence_len)
        return game, game.reset()

if __name__ == '__main__':
    unittest.main()