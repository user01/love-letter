# -*- coding: utf-8 -*-
"""
Love Letter Player object
Everything required to represent a player.
"""

from collections import namedtuple

import numpy as np

from loveletter.card import Card


# A record of an action taken during a turn.
# Card that was discarded and player that was targeted with the effect
#
#  discard - int card id discarded
#  player_target - int targeted player id
#  guess - int card guess being made - only useful if discard is 1 (guard), otherwise 0
#  revealed_card - int card of player_target - only useful if discard is 2 (priest), otherwise 0
#
# Note that the discarding player is a valid target and that is the only
# valid target for the non-effecting cards (ie handmaid or countess)
PlayerAction = namedtuple(
    'PlayerAction', 'discard player_target guess revealed_card')


class PlayerActionTools():
    """Functions to work with PlayerAction tuples"""

    @staticmethod
    def blank():
        """
        Generate a blank action (un-taken turn, either because the
        game is in progress or the player is knocked out
        """
        return PlayerAction(0, 0, 0, 0)

    @staticmethod
    def simple(card):
        """
        Generate an action to just discard, no effects
        """
        return PlayerAction(card, 0, 0, 0)

    @staticmethod
    def is_blank(action):
        """If an action is blank (ie empty)"""
        return action.discard == 0 and \
            action.player_target == 0 and \
            action.guess == 0 and \
            action.revealed_card == 0

    @staticmethod
    def from_np(arr):
        """Convert a player action tuple into a numpy array."""
        return PlayerAction(arr[0], arr[1], arr[2], arr[3])

    @staticmethod
    def to_np(player_action):
        """Convert a player action tuple into a numpy array."""
        return np.array(player_action, dtype=np.uint8)

    @staticmethod
    def from_np_many(player_actions):
        """Convert a player action tuple into a numpy array."""
        actions_split = np.reshape(
            player_actions, (player_actions.shape[0] // 4, 4))
        return list(map(PlayerActionTools.from_np, actions_split))

    @staticmethod
    def to_np_many(player_actions):
        """Convert a player action tuple into a numpy array."""
        actions = np.array(player_actions, dtype=np.uint8)
        return np.reshape(actions, len(player_actions) * 4)

# A Love Letter Player
#
#  hand_card - int corresponding to card currently in player's hand
#  actions - PlayerAction[] of all actions taken by the player
#            Note that actions are always of length 8 (the most)
#            number of moves a player can ever take in a game
#            in a 2 player game (technically, but highly unlikely)
Player = namedtuple('Player', 'hand_card actions')


class PlayerTools():
    """Functions to work with Player tuples"""

    @staticmethod
    def blank(dealt_card):
        """Generate a player with blank actions"""
        return Player(dealt_card, [PlayerActionTools.blank()] * 8)

    @staticmethod
    def move(player, hand_card_new, action):
        """Returns a new player object as the result of a move"""
        actions = player.actions[:]
        action_index = PlayerTools._next_empty_index(actions)
        actions[action_index] = action
        return Player(hand_card_new, actions)

    @staticmethod
    def set_hand(player, hand_card_new):
        """Returns a new player object as the result of a move"""
        return Player(hand_card_new, player.actions[:])

    @staticmethod
    def is_defended(player):
        """Returns if the player object is protected by a handmaid"""
        actions = player.actions[:]
        action_index = PlayerTools._next_empty_index(actions)
        if action_index == 0:
            return False

        last_discard = actions[action_index - 1].discard
        return last_discard == Card.handmaid

    @staticmethod
    def force_discard(player, new_card=Card.noCard):
        """Returns a new player object that is forced to discard"""
        return PlayerTools.move(player, new_card, PlayerActionTools.simple(player.hand_card))

    @staticmethod
    def _next_empty_index(actions):
        for idx, action in enumerate(actions):
            if action.discard == Card.noCard:
                return idx
        raise Exception("Insufficient space in actions")

    @staticmethod
    def is_playing(player):
        """Player still has a card"""
        return player.hand_card != 0

    @staticmethod
    def to_np(player):
        """Convert a player tuple into a numpy array."""
        hand_card = np.array([player.hand_card], dtype=np.uint8)
        actions = PlayerActionTools.to_np_many(player.actions)
        return np.concatenate([hand_card, actions])

    @staticmethod
    def from_np(arr):
        """Convert a player array into a Player tuple."""
        return Player(arr[0], PlayerActionTools.from_np_many(arr[1:]))
