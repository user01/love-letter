"""Tests for the running of the Love Letter game"""

import unittest
import numpy as np

from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools
from loveletter.tests.test_games import TestGames


class TestRemainingCards(unittest.TestCase):
    """Love Letter Games"""

    def test_consumed_cards(self):
        """Test if consumed cards is listed properly"""
        game = TestGames.replay(9, [3, 1, 0, 0])
        consumed_cards = game.consumed_cards()
        self.assertEqual(len(consumed_cards), 8)

        self.assertListEqual(list(consumed_cards),
                             [2 / 5,  # guards
                              0 / 2,  # priest
                              1 / 2,  # baron
                              0 / 2,  # handmaid
                              1 / 2,  # prince
                              0 / 1,  # king
                              0 / 1,  # countess
                              0 / 1])  # princess

    def test_consumed_cards_longer(self):
        """Test if consumed cards is listed properly in a longer game"""
        game = TestGames.replay(9, [3, 1, 0, 0, 1, 2, 2, 0, 6, 3,
                                    0, 0, 1, 2, 6, 0, 0, 0, 0, 0])
        consumed_cards = game.consumed_cards()
        self.assertEqual(len(consumed_cards), 8)

        self.assertListEqual(list(consumed_cards),
                             [3 / 5,  # guards
                              0 / 2,  # priest
                              1 / 2,  # baron
                              1 / 2,  # handmaid
                              1 / 2,  # prince
                              1 / 1,  # king
                              0 / 1,  # countess
                              0 / 1])  # princess


if __name__ == '__main__':
    unittest.main()
