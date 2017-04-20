"""Tests for the running of the Love Letter game"""

import unittest
import numpy as np

from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools
from loveletter.tests.test_games import TestGames

class TestRemainingCards(unittest.TestCase):
    """Love Letter Games"""

    def test_remaining_cards(self):
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2,
        6, 0, 7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0, 2, 1, 0, 0, 5, 0,
        0, 0])
        x = game.remaining_cards()
        self.assertEqual(len(x), 8)
        self.assertEqual(sum(x), 6.3)
        self.assertListEqual(list(x), [0.8, 1., 0.5, 1., 1., 1., 1., 0.])

    def test_state(self):
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2,
        6, 0, 7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0, 2, 1, 0, 0, 5, 0,
        0, 0])
        x = game.state()
        self.assertEqual(len(x), 24)
        self.assertEqual(sum(x), 8.3)
        self.assertListEqual(list(x), [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
        0., 0., 0., 0., 0., 0., 0.8, 1., 0.5, 1., 1., 1., 1., 0.])

if __name__ == '__main__':
    unittest.main()
