from operator import itemgetter

import gym
from gym import spaces
from gym.utils import seeding

from .game import Game
from .card import Card
from .player import PlayerAction, PlayerTools
from .agents.random import AgentRandom


class LoveLetterEnv(gym.Env):
    """Love Letter Game Environment

    The goal of hotter colder is to guess closer to a randomly selected number

    After each step the agent receives an observation of:
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target

    The rewards is calculated as:
    (min(action, self.number) + self.range) / (max(action, self.number) + self.range)

    Ideally an agent will be able to recognize the 'scent' of a higher reward and
    increase the rate in which is guesses in that direction until the reward reaches
    its maximum
    """

    def __init__(self, agent_other, seed=451):

        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24,))

        self._agent_other = AgentRandom(
            seed) if agent_other is None else agent_other
        self._seed(seed)
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        player_action = self.action_from_index(action)

        if player_action is None:
            return self._game.state(), -1, False, {"round": self._game.round()}

        self._game, reward = LoveLetterEnv.advance_game(
            self._game, player_action, self._agent_other)

        done = self._game.over() or not PlayerTools.is_playing(
            self._game.players()[0])

        return self._game.state(), reward, done, {"round": self._game.round()}

    def _reset(self):
        self._game = Game.new(4, self.np_random.random_integers(5000000))
        return self._game.state()

    def force(self, game):
        """Force the environment to a certain game state"""
        self._self = game
        return game.state()

    @staticmethod
    def advance_game(game, action, agent):
        """Advance a game with an action

        * Play an action
        * Advance the game using the agent
        * Return the game pending for the same player turn _unless_ the game ends

        returns <game, reward>
        """
        if not game.is_action_valid(action):
            return game, -1

        player_idx = game.player_turn()
        game_current, _ = game.move(action)
        while game_current.active():
            if not game_current.is_current_player_playing():
                game_current = game_current.skip_eliminated_player()
            elif game_current.player_turn() != player_idx:
                game_current, _ = game_current.move(agent.move(game_current))
            else:
                break

        # print("Round", game.round(), '->', game_current.round(), ':', 'OVER' if game_current.over() else 'RUNN')

        if game_current.over():
            if game_current.winner() == player_idx:
                return game_current, 15
            else:
                return game_current, -5

        return game_current, 0


    def action_by_score(self, scores, game=None):
        """
        Returns best action based on assigned scores

        return (action, score, idx)
        """
        if len(scores) != 15:
            raise Exception("Invalid scores length: {}".format(len(scores)))
        game = self._game if game is None else game

        assert game.active()
        actions_possible = self.actions_set(game)

        actions = [(action, score, idx) for action, score, idx in
                   zip(actions_possible,
                       scores,
                       range(len(actions_possible)))
                   if game.is_action_valid(action)]

        action = max(actions, key=itemgetter(2))
        return action

    def action_from_index(self, action_index, game=None):
        """Returns valid (idx, actions) based on a current game"""
        game = self._game if game is None else game

        action_candidates = self.actions_set(game)

        actions = [(idx, action) for idx, action in
                   enumerate(action_candidates)
                   if game.is_action_valid(action) and idx == action_index]

        return actions[0][1] if len(actions) == 1 else None

    def actions_possible(self, game=None):
        """Returns valid (idx, actions) based on a current game"""
        game = self._game if game is None else game

        action_candidates = self.actions_set(game)

        actions = [(idx, action) for idx, action in
                   enumerate(action_candidates)
                   if game.is_action_valid(action)]

        return actions

    def actions_set(self, game=None):
        """Returns all actions for a game"""
        game = self._game if game is None else game

        player_self = game.player_turn()
        opponents = game.opponent_turn()

        actions_possible = [
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.priest,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.baron,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.handmaid,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.prince,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.king,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.countess,
                         Card.noCard),
            PlayerAction(Card.guard,
                         self.np_random.choice(opponents),
                         Card.princess,
                         Card.noCard),
            PlayerAction(Card.priest,
                         self.np_random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.baron,
                         self.np_random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.king,
                         self.np_random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.prince,
                         self.np_random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.prince, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.handmaid, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.countess, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.princess, player_self, Card.noCard, Card.noCard)
        ]

        return actions_possible
