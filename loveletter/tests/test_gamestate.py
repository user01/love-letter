"""Testing the game state."""

import unittest
import numpy as np

from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools
from loveletter.tests.test_games import TestGames


class TestGamestate(unittest.TestCase):
    """Love Letter Games"""

    def test_state_hand(self):
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2, 6, 0,
                                    7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0,
                                    2, 1, 0, 0, 5, 0, 0, 0])
        result = game.state_hand()
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 8)
        self.assertEqual(len(result[1]), 8)
        self.assertEqual(sum(result[0]), 1)
        self.assertEqual(sum(result[1]), 1)
        self.assertListEqual(list(result[0]), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertListEqual(list(result[1]), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # card = index number of array

if __name__ == '__main__':
    unittest.main()
