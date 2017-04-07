"""Tests for the main Love Letter game"""

import unittest
from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools
from loveletter.player import PlayerTools


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


class TestBasic(unittest.TestCase):
    """Test of Basic Game operations"""

    def test_new(self):
        """Getting a new game"""
        game = Game.new()
        self.assertEqual(game.draw_card(), Card.guard)
        self.assertEqual(game.round(), 0)
        self.assertEqual(game.player_turn(), 0)
        self.assertEqual(game.cards_left(), 10)
        self.assertTrue(game.active())
        self.assertFalse(game.over())

    def test_move_guard_failure(self):
        """Getting a guard move, with a wrong guess"""
        game = Game.new()
        action = PlayerAction(Card.guard, 1, Card.handmaid, 0)
        game = game.move(action)

        self.assertEqual(game.round(), 0)
        self.assertEqual(game.player_turn(), 1)
        self.assertEqual(game.cards_left(), 9)
        self.assertTrue(game.active())
        self.assertFalse(game.over())

        players = game.players()
        player = players[0]
        target = players[1]
        recent_action = player.actions[0]

        self.assertTrue(PlayerTools.is_playing(target))
        self.assertEqual(recent_action, action)
        self.assertEqual(player.hand_card, Card.handmaid)
        self.assertFalse(PlayerActionTools.is_blank(recent_action))
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))


    def test_move_guard_success(self):
        """Getting a guard move, with a right guess"""
        game = Game.new()
        action = PlayerAction(Card.guard, 3, Card.handmaid, 0)
        game = game.move(action)

        self.assertEqual(game.round(), 0)
        self.assertEqual(game.player_turn(), 1)
        self.assertEqual(game.cards_left(), 9)
        self.assertTrue(game.active())
        self.assertFalse(game.over())

        players = game.players()
        player = players[0]
        target = players[3]
        recent_action = player.actions[0]

        self.assertFalse(PlayerTools.is_playing(target))
        self.assertEqual(recent_action, action)
        self.assertEqual(player.hand_card, Card.handmaid)
        self.assertFalse(PlayerActionTools.is_blank(recent_action))
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_priest(self):
        """Getting a priest move"""
        game = Game.new(4, 5)
        action = PlayerAction(Card.priest, 1, Card.noCard, Card.noCard)
        action_expected = PlayerAction(Card.priest, 1, Card.noCard, Card.guard)
        game = game.move(action)

        self.assertEqual(game.round(), 0)
        self.assertEqual(game.player_turn(), 1)
        self.assertEqual(game.cards_left(), 9)
        self.assertTrue(game.active())
        self.assertFalse(game.over())

        players = game.players()
        player = players[0]
        target = players[1]
        recent_action = player.actions[0]

        self.assertTrue(PlayerTools.is_playing(target))
        self.assertEqual(recent_action, action_expected)
        self.assertFalse(PlayerActionTools.is_blank(recent_action))
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_baron_success(self):
        """Getting a baron move, with a success"""
        game = Game.new(4, 48)
        action = PlayerAction(Card.baron, 3, Card.noCard, Card.noCard)
        game = game.move(action)

        players = game.players()
        player = players[0]
        target = players[3]
        recent_action = player.actions[0]

        self.assertTrue(PlayerTools.is_playing(player))
        self.assertFalse(PlayerTools.is_playing(target))
        self.assertEqual(recent_action, action)

        self.assertFalse(PlayerActionTools.is_blank(recent_action))
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

        self.assertFalse(PlayerActionTools.is_blank(target.actions[0]))
        for action in target.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_baron_failure(self):
        """Getting a baron move, with a failure"""
        game = Game.new(4, 48)
        action = PlayerAction(Card.baron, 1, Card.noCard, Card.noCard)
        game = game.move(action)

        players = game.players()
        player = players[0]
        target = players[1]

        self.assertFalse(PlayerTools.is_playing(player))
        self.assertTrue(PlayerTools.is_playing(target))

        self.assertFalse(PlayerActionTools.is_blank(player.actions[0]))
        self.assertFalse(PlayerActionTools.is_blank(player.actions[1]))
        for action in player.actions[2:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

        for action in target.actions:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_handmaid(self):
        """Deploy the handmaid and survive attack"""
        game = Game.new(4, 2)
        action = PlayerAction(Card.handmaid, 0, Card.noCard, Card.noCard)
        game = game.move(action)

        players = game.players()
        player = players[0]
        self.assertTrue(PlayerTools.is_playing(player))
        self.assertFalse(PlayerActionTools.is_blank(player.actions[0]))
        self.assertEqual(player.actions[0], action)
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

        action_attack = PlayerAction(Card.guard, 0, Card.prince, Card.noCard)
        game = game.move(action_attack)

        players = game.players()
        target = players[0]
        player = players[1]
        self.assertTrue(PlayerTools.is_playing(player))
        self.assertTrue(PlayerTools.is_playing(target))

        self.assertFalse(PlayerActionTools.is_blank(player.actions[0]))
        self.assertEqual(player.actions[0], action_attack)
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

        for action in target.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_prince_self(self):
        """Use prince to force self discard"""
        game = Game.new(4, 2)
        action = PlayerAction(Card.prince, 0, Card.noCard, Card.noCard)
        action_other = PlayerAction(Card.handmaid, 0, Card.noCard, Card.noCard)
        game = game.move(action)

        players = game.players()
        player = players[0]
        self.assertTrue(PlayerTools.is_playing(player))
        self.assertFalse(PlayerActionTools.is_blank(player.actions[0]))
        self.assertFalse(PlayerActionTools.is_blank(player.actions[1]))
        self.assertEqual(player.actions[0], action)
        self.assertEqual(player.actions[1], action_other)
        for action in player.actions[2:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

    def test_move_prince_other(self):
        """Use prince to force another to discard"""
        game = Game.new(4, 2)
        action = PlayerAction(Card.prince, 1, Card.noCard, Card.noCard)
        action_target = PlayerAction(Card.guard, 0, Card.noCard, Card.noCard)
        game = game.move(action)

        players = game.players()
        player = players[0]
        target = players[1]

        self.assertTrue(PlayerTools.is_playing(player))
        self.assertFalse(PlayerActionTools.is_blank(player.actions[0]))
        self.assertEqual(player.actions[0], action)
        for action in player.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))

        self.assertTrue(PlayerTools.is_playing(target))
        self.assertFalse(PlayerActionTools.is_blank(target.actions[0]))
        self.assertEqual(target.actions[0], action_target)
        for action in target.actions[1:]:
            self.assertTrue(PlayerActionTools.is_blank(action))



if __name__ == '__main__':
    unittest.main()
