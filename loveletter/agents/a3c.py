"""Agent with uses A3C trained network"""

import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable


from loveletter.env import LoveLetterEnv
from loveletter.agents.random import AgentRandom
from loveletter.agents.agent import Agent
from loveletter.trainers.a3c_model import ActorCritic



class AgentA3C(Agent):
    '''Agent which leverages Actor Critic Learning'''

    def __init__(self,
                 model_path,
                 dtype,
                 seed=451):
        self._seed = seed
        self._idx = 0
        self._dtype = dtype
        self.env = LoveLetterEnv(AgentRandom(seed), seed)
        state = self.env.reset()

        self._model = ActorCritic(
            state.shape[0], self.env.action_space).type(dtype)
        self._model.load_state_dict(torch.load(model_path))

    def _move(self, game):
        '''Return move which ends in score hole'''
        assert game.active()
        self._idx += 1

        state = self.env.force(game)
        state = torch.from_numpy(state).type(self._dtype)
        cx = Variable(torch.zeros(1, 256).type(self._dtype), volatile=True)
        hx = Variable(torch.zeros(1, 256).type(self._dtype), volatile=True)

        _, logit, (hx, cx) = self._model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action_idx = prob.max(1)[1].data.cpu().numpy()[0, 0]

        player_action = self.env.action_from_index(action_idx, game)
        if player_action is None:
            # print("ouch")
            options = Agent.valid_actions(game, self._seed + self._idx)
            if len(options) < 1:
                raise Exception("Unable to play without actions")

            random.seed(self._seed + self._idx)
            return random.choice(options)

        # print("playing ", self._idx, player_action)
        return player_action
