# -*- coding: utf-8 -*-
"""
Love Letter Game object
"""

from card import Card
from player import Player, PlayerTools, PlayerAction, PlayerActionTools


class Game():
    """A Love Letter Game"""

    def __init__(self, deck, players, player_turn):
        self._deck = deck
        self._players = players
        self._player_turn = player_turn

    def players(self):
        """List of current players."""
        return self._players[:]

    def deck(self):
        """List of current cards."""
        return self._deck[:-1]

    def card_hidden(self):
        """Cards hidden from play."""
        return self._deck[-1]

    def player_turn(self):
        """Player number of current player."""
        return self._player_turn

    @staticmethod
    def new(player_count=4, seed=451):
        """Create a brand new game"""
        deck = Card.shuffle_deck(seed)

        dealt_cards = deck[:player_count]
        undealt_cards = deck[player_count:]

        players = list(map(lambda card: Player(card, []), dealt_cards))
        return Game(undealt_cards, players, 0)
