# -*- coding: utf-8 -*-
"""
Love Letter Game object
"""


class Game():
    """A Love Letter Game"""

    def __init__(self, deck, players):
        self._deck = deck
        self._players = players

    def players(self):
        """List of current players."""
        return self._players[:]

    @staticmethod
    def new(seed=451):
        """Create a brand new game"""
        return seed
