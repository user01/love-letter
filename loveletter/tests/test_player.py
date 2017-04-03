"""Tests for Love Letter Player tools"""

import unittest
import numpy as np
from loveletter.player import Player, PlayerTools
from loveletter.player import PlayerAction, PlayerActionTools


class TestPlayer(unittest.TestCase):
    """Test Player Tools for the Love Letter Game"""

    def test_init(self):
        """Create a Player"""
        player = Player(1, [])  # hand card and past actions
        self.assertEqual(player.hand_card, 1)
        self.assertEqual(player.actions, [])

    def test_to_np(self):
        """Player to a numpy array"""
        player = Player(1, [
            PlayerAction(1, 3, 5, 0),
            PlayerAction(3, 0, 0, 0)
        ])
        arr = np.array([1, 1, 3, 5, 0, 3, 0, 0, 0], dtype=np.uint8)
        arr_res = PlayerTools.to_np(player)
        self.assertEqual(len(arr_res), len(arr))
        self.assertTrue((arr_res == arr).all())

    def test_from_np(self):
        """Player from a numpy array"""
        player = Player(1, [
            PlayerAction(1, 3, 5, 0),
            PlayerAction(3, 0, 0, 2)
        ])
        arr = np.array([1, 1, 3, 5, 0, 3, 0, 0, 2], dtype=np.uint8)
        player_res = PlayerTools.from_np(arr)
        self.assertEqual(player_res, player)


class TestPlayerActions(unittest.TestCase):
    """Test Player actions tools for the Love Letter Game"""

    def test_init(self):
        """Create an action"""
        # play guard, on player 3, guessing Prince, no revealed card
        action = PlayerAction(1, 3, 5, 0)
        self.assertEqual(action.discard, 1)
        self.assertEqual(action.player_target, 3)
        self.assertEqual(action.guess, 5)
        self.assertEqual(action.revealed_card, 0)
        self.assertEqual(PlayerActionTools.is_blank(action), False)

    def test_blank(self):
        """Create an action"""
        # play guard, on player 3, guessing Prince, no revealed card
        action_normal = PlayerAction(1, 3, 5, 0)
        self.assertEqual(PlayerActionTools.is_blank(action_normal), False)
        # no action - either hasn't been taken or player was eliminated
        action_blank = PlayerAction(0, 0, 0, 0)
        self.assertEqual(PlayerActionTools.is_blank(action_blank), True)

    def test_to_np(self):
        """Action to a numpy array"""
        action = PlayerAction(1, 3, 5, 0)
        arr = np.array([1, 3, 5, 0], dtype=np.uint8)
        self.assertTrue((PlayerActionTools.to_np(action) == arr).all())

    def test_from_np(self):
        """Action from a numpy array"""
        arr = np.array([1, 3, 5, 0], dtype=np.uint8)
        action = PlayerActionTools.from_np(arr)
        self.assertEqual(action.discard, 1)
        self.assertEqual(action.player_target, 3)
        self.assertEqual(action.guess, 5)
        self.assertEqual(action.revealed_card, 0)

    def test_to_np_many(self):
        """Actions to a numpy array"""
        actions = [
            PlayerAction(1, 3, 5, 0),
            PlayerAction(4, 0, 5, 0),
            PlayerAction(8, 0, 0, 0),
            PlayerAction(5, 0, 0, 0)
        ]
        arr = np.array([1, 3, 5, 0,
                        4, 0, 5, 0,
                        8, 0, 0, 0,
                        5, 0, 0, 0], dtype=np.uint8)
        self.assertTrue((PlayerActionTools.to_np_many(actions) == arr).all())

    def test_from_np_many(self):
        """Action from a numpy array"""
        actions = [
            PlayerAction(1, 3, 5, 1),
            PlayerAction(4, 0, 5, 1),
            PlayerAction(8, 0, 0, 1),
            PlayerAction(5, 0, 0, 1)
        ]
        arr = np.array([1, 3, 5, 1,
                        4, 0, 5, 1,
                        8, 0, 0, 1,
                        5, 0, 0, 1], dtype=np.uint8)
        self.assertListEqual(PlayerActionTools.from_np_many(arr), actions)


if __name__ == '__main__':
    unittest.main()
