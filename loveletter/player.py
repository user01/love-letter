# -*- coding: utf-8 -*-
"""
Love Letter Player object
Everything required to represent a player.
"""

import numpy as np
from collections import namedtuple

# A record of an action taken during a turn.
# Card that was discarded and player that was targeted with the effect
#
# Note that the discarding player is a valid target and that is the only
# valid target for the non-effecting cards (ie handmaid or countess)
PlayerAction = namedtuple('PlayerAction', 'discard player_target guess')
class PlayerActionTools():

    @staticmethod
    def to_np(player_action):
        """Convert a player action tuple into a numpy array."""
        return np.array(player_action, dtype=np.uint8)


# A Love Letter Player
Player = namedtuple('Player', 'hand_card actions')
class PlayerTools():

    @staticmethod
    def is_playing(player):
        """Player still has a card"""
        return player._hand_card != 0
