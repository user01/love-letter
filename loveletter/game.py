# -*- coding: utf-8 -*-
"""
Love Letter Game object
"""

from card import shuffle_deck
from player import Player

class Game():
    """A Love Letter Game"""

    def __init__(self, deck, players):
        self._deck = deck
        self._players = players

    def players(self):
        """List of current players."""
        return self._players[:]

    @staticmethod
    def new(player_count=4, seed=451):
        """Create a brand new game"""
        deck = shuffle_deck(seed)

        dealt_cards = deck[:player_count]
        undealt_cards = deck[player_count:]

        players = list(map(lambda x: Player(x, []), dealt_cards))
        return Game(undealt_cards, players)
