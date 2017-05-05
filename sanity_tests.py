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
        self.assertEquals(obv[0]["INDEX"], 1)
        self.assertEquals(reward, 1)
        self.assertEquals(done, True)



    def setup(self, num_pairs=2, noise_per_person=1, num_channels=4, sequence_len=5):
        game = Spectrum()
        game.set_constants(num_pairs, noise_per_person, num_channels, sequence_len)
        return game, game.reset()

if __name__ == '__main__':
    unittest.main()