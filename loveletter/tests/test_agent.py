"""Testing the abstract agent"""

import unittest

from loveletter.agents.agent import Agent
from loveletter.card import Card
from loveletter.tests.test_games import TestGames
from loveletter.player import PlayerAction


class TestAbstractAgent(unittest.TestCase):
    """Love Letter Abstract Agent"""

    def test_valid_actions(self):
        """Test if the actions generated for a player at a game state is valid."""
        game = TestGames.replay(1, [1, 1, 3, 0, 4, 1, 0, 0, 1, 1, 3, 0, 1, 2, 6, 0,
                                    7, 0, 0, 0, 1, 2, 2, 0, 8, 2, 0, 0, 1, 1, 5, 0])
        actions = Agent.valid_actions(game, 782)
        self.assertEqual(len(actions), 2)

        # player can use baron or priest, at 1 or 3

        self.assertListEqual(actions,
                             [PlayerAction(discard=Card.priest,
                                           player_target=3,
                                           guess=0,
                                           revealed_card=0),
                              PlayerAction(discard=Card.baron,
                                           player_target=3,
                                           guess=0,
                                           revealed_card=0)])


if __name__ == '__main__':
    unittest.main()
