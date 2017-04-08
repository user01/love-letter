"""Tests for the running of the Love Letter game"""

import unittest
import numpy as np

from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools


class TestGames(unittest.TestCase):
    """Love Letter Games"""

    @staticmethod
    def replay(seed, action_sequence=None):
        """Generate a game from a recorded set of actions"""
        action_sequence = action_sequence if action_sequence is not None else []
        action_sequence = np.array(action_sequence, dtype=np.uint8)
        action_sequence = PlayerActionTools.from_np_many(action_sequence)[::-1]
        game = Game.new(4, seed)

        while len(action_sequence) > 0:
            if not game.is_current_player_playing():
                game = game.skip_eliminated_player()
            else:
                action = action_sequence.pop()
                game = game.move(action)

        return game

    def test_end_elimination(self):
        """Reach the end of a game by elimination"""
        game = Game.new(4, 0)

        game = game.move(PlayerAction(
            Card.guard, 1, Card.priest, Card.noCard))
        game = game.skip_eliminated_player()
        game = game.move(PlayerAction(
            Card.guard, 3, Card.countess, Card.noCard))
        game = game.move(PlayerAction(
            Card.handmaid, 3, Card.noCard, Card.noCard))

        game = game.move(PlayerAction(
            Card.countess, 0, Card.noCard, Card.noCard))
        game = game.skip_eliminated_player()
        game = game.move(PlayerAction(
            Card.baron, 3, Card.noCard, Card.noCard))
        game = game.move(PlayerAction(
            Card.guard, 2, Card.handmaid, Card.noCard))

        self.assertFalse(game.over())
        game = game.move(PlayerAction(
            Card.princess, 0, Card.noCard, Card.noCard))

        self.assertEqual(game.cards_left(), 4)
        self.assertTrue(game.over())

    def test_end_length(self):
        """Reach the end of a game by cards"""
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2, 6, 0,
                                    7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0,
                                    2, 1, 0, 0, 5, 0, 0, 0])

        self.assertEqual(game.cards_left(), 0)
        self.assertTrue(game.over())


if __name__ == '__main__':
    unittest.main()
