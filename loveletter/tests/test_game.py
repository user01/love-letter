"""Tests for the main Love Letter game"""

import unittest
from loveletter.game import Game


class TestInit(unittest.TestCase):
    """Initialization of the Love Letter Game"""

    def test_addition(self):
        """Example Test"""
        self.assertEqual(1 + 0, 1)


class TestStatics(unittest.TestCase):
    """Test of Game static functions"""

    def test_new_hand_card(self):
        """Getting a new hand card post discard"""
        self.assertEqual(Game.new_hand_card(1, [1, 2]), 2)
        self.assertEqual(Game.new_hand_card(2, [1, 2]), 1)
        self.assertEqual(Game.new_hand_card(2, [2, 2]), 2)


if __name__ == '__main__':
    unittest.main()
