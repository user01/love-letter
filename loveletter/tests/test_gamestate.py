"""Testing the game state."""

import unittest
import numpy as np


from loveletter.card import Card
from loveletter.tests.test_games import TestGames


class TestGamestate(unittest.TestCase):
    """Love Letter Games"""


    def test_game_state(self):
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2, 6, 0,
                                    7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0,
                                    2, 1, 0, 0])
        state = game.state()

        self.assertEqual(len(state), 24)
        self.check_hot_encoded(state[0:8], Card.prince)
        self.check_hot_encoded(state[8:16], Card.king)

        self.assertListEqual(list(state[16:24]),
                             [5 / 5,  # guards
                              1 / 2,  # priest
                              1 / 2,  # baron
                              1 / 2,  # handmaid
                              1 / 2,  # prince
                              1 / 1,  # king
                              1 / 1,  # countess
                              1 / 1])  # princess

    def check_hot_encoded(self, np_arr, card_idx):
        """Checks that all indices except for the target are 0"""
        for idx in range(8):
            # NOTE: since the hand can never have no card, the 0th noCard is never used
            # as such, it is eliminated and all indices are pushed back one
            if idx == card_idx - 1:
                self.assertEqual(np_arr[idx], 1)
            else:
                self.assertEqual(np_arr[idx], 0)


if __name__ == '__main__':
    unittest.main()
