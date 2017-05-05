import numpy as np
from loveletter.env import LoveLetterEnv
from loveletter.agents.agent import Agent
from loveletter.agents.random import AgentRandom
import random

class AgentSarsaLambda(Agent):
    '''Agent which leverages Sarsa Lambda Learning'''

    def __init__(self, seed = 451,
                q_path = 'models/sarsa_lambda_Q.npz'):
        with np.load(q_path) as data:
            self._Q= data['Q']

        self._seed = seed
        self._idx = 0
        self.env = LoveLetterEnv(AgentRandom(seed), seed)     

    def _nn_to_ss(self, state):
        lh = np.argmax(state[0:8])
        rh = np.argmax(state[8:16])
        cards = np.multiply(state[16:24], [5, 2, 2, 2, 2, 1, 1, 1]).astype(int)

        return np.concatenate(([lh, rh], cards))   

    def _move(self, game):
        assert game.active()

        S = self._nn_to_ss(game.state())

        A = np.argmax(self._Q[:, tuple(S)])

        player_action = self.env.action_from_index(A, game)
        #print(player_action)

        if player_action is None:
            # print("ouch")
            options = Agent.valid_actions(game, self._seed + self._idx)
            if len(options) < 1:
                raise Exception("Unable to play without actions")

            random.seed(self._seed + self._idx)
            return random.choice(options)

        # print("playing ", self._idx, player_action)
        return player_action

