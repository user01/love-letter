# -*- coding: utf-8 -*-
"""
Love Letter Card tools
Functions and constants to facilitate working with cards, which are represented as integers.
"""

import numpy as np


class Card():
    """Static Card class"""
    noCard = 0
    guard = 1
    priest = 2
    baron = 3
    handmaid = 4
    prince = 5
    king = 6
    countess = 7
    princess = 8

    #              0          1        2         3
    names = ['', 'Guard', 'Priest', 'Baron',
             # 4           5         6       7           8
             'Handmaid', 'Prince', 'King', 'Countess', 'Princess']
    #           0     1     2    3     4    5    6    7    8
    symbols = ['☁️', '⚔️', '🕌', '🎲', '🛡️', '⚜️', '👑', '👸', '❤️']

    descriptions = ['None',  # None
                    'Guess a player\'s hand',  # Guard
                    'Look at a hand',  # Priest
                    'Compare hands; lower hand is out.',  # Baron
                    'Protection until your next turn',  # Handmaid
                    'One player discards their hand',  # Prince
                    'Trade hands with target player',  # King
                    'Discard if caught with King or Prince',  # Countess
                    'Lose if discarded']  # Princess

    counts = [5,  # Guard
              2,  # Priest
              2,  # Baron
              2,  # Handmaid
              1,  # Prince
              1,  # King
              1,  # Countess
              1]  # Princess

    only_self = [4, 7, 8]
    only_other = [1, 2, 3, 6]

    @staticmethod
    def render_card_number(card):
        """Render a card name with padded length"""
        numbered_names = ["{} {} ({})".format(name, symbol, idx)
                          for idx, (name, symbol) in enumerate(zip(Card.names, Card.symbols))]
        max_length = max([len(i) for i in numbered_names])
        str_base = "{0: >" + str(max_length) + "}"
        return str_base.format(numbered_names[card])

    @staticmethod
    def shuffle_deck(seed=451):
        """A numpy array of shuffled cards"""
        deck = []
        for card_number, card_count in enumerate(Card.counts):
            card_id = card_number + 1
            deck = deck + [card_id] * card_count
        deck_np = np.array(deck)
        np.random.seed(seed=seed)
        np.random.shuffle(deck_np)
        return deck_np
