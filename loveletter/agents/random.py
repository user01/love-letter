# -*- coding: utf-8 -*-

"""
Random Agent for the a Love Letter AI
"""

import random
from .agent import Agent


class AgentRandom(Agent):
    """Random Player Class for play Love Letter."""

    def __init__(self, seed=451):
        self._seed = seed
        self._idx = 0

    def _move(self, game):
        """Return a random valid move"""
        self._idx = self._idx + 1
        seed = self._seed + self._idx
        random.seed(seed)

        options = Agent.valid_actions(game, seed)
        if len(options) < 1:
            raise Exception("Unable to play without actions")

        return random.choice(options)
