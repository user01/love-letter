# -*- coding: utf-8 -*-
"""
Love Letter Player object
Everything required to represent a player.
"""


class Player():
    """A Love Letter Player"""

    def __init__(self, hand_card, actions):
        self._hand_card = hand_card
        self._actions = actions

    def is_playing(self):
        """Player still has a card"""
        return self._hand_card != 0

    def actions(self):
        """List of current player actions taken by this player."""
        return self._actions[:]


class PlayerAction():
    """
    A record of an action taken during a turn.
    Note this can be null

    TODO: Move this to a tuple
    """

    def __init__(self, discard, player_target):
        """
        Card that was discarded and player that was targeted with the effect

        Note that the discarding player is a valid target and that is the only
        valid target for the non-effecting cards (ie handmaid or countess)
        """
        self._discard = discard
        self._player_target = player_target

    def discard(self):
        """Discarded card"""
        return self._discard

    def player_target(self):
        """Player targeted with action"""
        return self._player_target

    def is_no_action(self):
        """If the action is none"""
        return self._discard == 0
