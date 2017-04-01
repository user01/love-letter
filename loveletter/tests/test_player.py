"""Tests for Love Letter Player tools"""

import unittest
import numpy as np
from loveletter.player import PlayerAction, PlayerActionTools

class TestPlayerActions(unittest.TestCase):
    """Test Player actions tools for the Love Letter Game"""

    def test_init(self):
        """Create an action"""
        action = PlayerAction(1, 3, 5) # play guard, on player 3, guessing Prince
        self.assertEqual(action.discard, 1)
        self.assertEqual(action.player_target, 3)
        self.assertEqual(action.guess, 5)

    def test_to_np(self):
        """Action to a numpy array"""
        action = PlayerAction(1, 3, 5)
        arr = np.array([1, 3, 5], dtype=np.uint8)
        self.assertTrue((PlayerActionTools.to_np(action) == arr).all())

    def test_from_np(self):
        """Action from a numpy array"""
        arr = np.array([1, 3, 5], dtype=np.uint8)
        action = PlayerActionTools.from_np(arr)
        self.assertEqual(action.discard, 1)
        self.assertEqual(action.player_target, 3)
        self.assertEqual(action.guess, 5)

    def test_to_np_many(self):
        """Actions to a numpy array"""
        actions = [
            PlayerAction(1, 3, 5),
            PlayerAction(4, 0, 5),
            PlayerAction(8, 0, 0),
            PlayerAction(5, 0, 0)
        ]
        arr = np.array([1, 3, 5,
                        4, 0, 5,
                        8, 0, 0,
                        5, 0, 0], dtype=np.uint8)
        self.assertTrue((PlayerActionTools.to_np_many(actions) == arr).all())

    def test_from_np(self):
        """Action from a numpy array"""
        actions = [
            PlayerAction(1, 3, 5),
            PlayerAction(4, 0, 5),
            PlayerAction(8, 0, 0),
            PlayerAction(5, 0, 0)
        ]
        arr = np.array([1, 3, 5,
                        4, 0, 5,
                        8, 0, 0,
                        5, 0, 0], dtype=np.uint8)
        self.assertListEqual(PlayerActionTools.from_np_many(arr), actions)


if __name__ == '__main__':
    unittest.main()
