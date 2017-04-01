# -*- coding: utf-8 -*-
"""
Love Letter Player object
Everything required to represent a player.
"""

from collections import namedtuple

import numpy as np

# A record of an action taken during a turn.
# Card that was discarded and player that was targeted with the effect
#
#  discard - int card id discarded
#  player_target - int targeted player id
#  guess - int card guess being made - only useful if discard is 1 (guard), otherwise 0
#
# Note that the discarding player is a valid target and that is the only
# valid target for the non-effecting cards (ie handmaid or countess)
PlayerAction = namedtuple(
    'PlayerAction', 'discard player_target guess')


class PlayerActionTools():
    """Functions to work with PlayerAction tuples"""

    @staticmethod
    def from_np(arr):
        """Convert a player action tuple into a numpy array."""
        return PlayerAction(arr[0], arr[1], arr[2])

    @staticmethod
    def to_np(player_action):
        """Convert a player action tuple into a numpy array."""
        return np.array(player_action, dtype=np.uint8)

    @staticmethod
    def from_np_many(player_actions):
        """Convert a player action tuple into a numpy array."""
        actions_split = np.reshape(
            player_actions, (player_actions.shape[0] // 3, 3))
        return list(map(PlayerActionTools.from_np, actions_split))

    @staticmethod
    def to_np_many(player_actions):
        """Convert a player action tuple into a numpy array."""
        actions = np.array(player_actions, dtype=np.uint8)
        return np.reshape(actions, len(player_actions) * 3)

# A Love Letter Player
#
#  hand_card - int corresponding to card currently in player's hand
#  actions - PlayerAction[] of all actions taken by the player
Player = namedtuple('Player', 'hand_card actions')
class PlayerTools():
    """Functions to work with Player tuples"""

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
        return Player(arr[0], PlayerActionTools.from_np_many(arr[0:]))
