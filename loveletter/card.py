# -*- coding: utf-8 -*-
"""
Love Letter Card tools
Functions and constants to facilitate working with cards, which are represented as integers.
"""

import numpy as np

CARD_NAMES = ['No Card', 'Guard', 'Priest', 'Baron',
              'Handmaid', 'Prince', 'King', 'Countess', 'Princess']

CARD_DESCRIPTIONS = ['None',
                     'Guess a player\'s hand',
                     'Look at a hand',
                     'Compare hands; lower hand is out.',
                     'Protection until your next turn',
                     'One player discards their hand',
                     'Trade hands',
                     'Discard if caught with King or Prince',
                     'Lose if discarded']

CARD_COUNTS = [5,  # Guard
               2,  # Priest
               2,  # Baron
               2,  # Handmaid
               1,  # Prince
               1,  # King
               1,  # Countess
               1]  # Princess


def shuffle_deck():
    """A numpy array of shuffled cards"""
    deck = []
    for card_number, card_count in enumerate(CARD_COUNTS):
        card_id = card_number + 1
        deck = deck + [card_id] * card_count
    deck_np = np.array(deck)
    np.random.shuffle(deck_np)
    return deck_np
