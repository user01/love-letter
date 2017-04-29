# -*- coding: utf-8 -*-

"""
Deep Q Learning Agent for the a Love Letter AI
"""

import random
import math
import time
import sys
from collections import namedtuple
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

from loveletter.game import Game
from loveletter.card import Card
from loveletter.player import PlayerAction
from .agent import Agent
from .random import AgentRandom


class AgentDQN(Agent):
    '''
    Agent which leverages Deep Q Learning
    '''

    def __init__(self,
                 model_path=None,
                 seed=451):
        self._seed = seed
        self._idx = 0
        self._model = ModelDQN()
        ModelDQN.Seed(seed)

        if ModelDQN.USE_CUDA:
            self._model.cuda()

        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path))

    def _move(self, game):
        '''Return move which ends in score hole'''
        self._idx = self._idx + 1
        seed = self._seed + self._idx
        ModelDQN.Seed(seed)
        state = torch.from_numpy(game.state()).type(torch.FloatTensor)

        scores = self._model(
            ModelDQN.Variable(state.unsqueeze(0))
        ).data.cpu().tolist()[0]
        action, _, _ = AgentDQN.action_by_score(game, scores, seed)

        return action

    @staticmethod
    def action_by_score(game, scores, seed=451):
        """
        Returns best action based on assigned scores

        return (action, score, idx)
        """
        if len(scores) != 15:
            raise Exception("Invalid scores length")

        actions_possible = AgentDQN.actions_set(game, seed)

        actions = [(action, score, idx) for action, score, idx in
                   zip(actions_possible,
                       scores,
                       range(len(actions_possible)))
                   if game.is_action_valid(action)]

        action = max(actions, key=itemgetter(2))
        return action

    @staticmethod
    def actions_possible(game, seed=451):
        """Returns valid (idx, actions) based on a current game"""

        action_candidates = AgentDQN.actions_set(game, seed)

        actions = [(idx, action) for idx, action in
                   enumerate(action_candidates)
                   if game.is_action_valid(action)]

        return actions

    @staticmethod
    def actions_set(game, seed=451):
        """Returns all actions for a game"""

        random.seed(seed + game.round())
        player_self = game.player_turn()
        opponents = game.opponent_turn()

        actions_possible = [
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.priest,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.baron,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.handmaid,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.prince,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.king,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.countess,
                         Card.noCard),
            PlayerAction(Card.guard,
                         random.choice(opponents),
                         Card.princess,
                         Card.noCard),
            PlayerAction(Card.priest,
                         random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.baron,
                         random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.king,
                         random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.prince,
                         random.choice(opponents),
                         Card.noCard,
                         Card.noCard),
            PlayerAction(Card.prince, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.handmaid, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.countess, player_self, Card.noCard, Card.noCard),
            PlayerAction(Card.princess, player_self, Card.noCard, Card.noCard)
        ]

        return actions_possible


class TrainerDQN():
    """
    Training class for simple Deep Q Learning

    Based on
    http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self,
                 model_path=None,
                 seed=451,
                 batch_size=128,
                 gamma=0.999,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 replay_size=10000,
                 learning_rate=0.02):
        self._run = 0
        self._steps_done = 0
        self._seed = seed

        self._BATCH_SIZE = batch_size
        self._GAMMA = gamma
        self._EPS_START = eps_start
        self._EPS_END = eps_end
        self._EPS_DECAY = eps_decay
        ModelDQN.Seed(seed)

        self._model = ModelDQN()
        self._memory = ReplayMemory(replay_size)
        self._optimizer = optim.RMSprop(
            self._model.parameters(), lr=learning_rate)

        if ModelDQN.USE_CUDA:
            self._model.cuda()
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path))

    def write_state_to_path(self, path):
        torch.save(self._model.state_dict(), path)

    def train(self, num_episodes=10, agent=None, print_mod=1):
        self._run += 1
        actions = 0
        self._steps_done = 0
        time_start = time.time()
        time_last = time_start
        agent = AgentRandom(self._seed + self._run) if agent is None else agent

        for idx in range(num_episodes):
            # Initialize the environment and state
            if idx % print_mod == 0 or time.time() - time_last > 30:
                sys.stdout.write(
                    " steps:{:0>5}/episode:{:0>5}/memory:{:0>5}:".format(self._steps_done,
                                                    idx,
                                                    len(self._memory)))
                sys.stdout.flush()
                time_last = time.time()
            game = Game.new(4, idx)

            while game.active():
                # Select and perform an action
                actions += 1
                action, action_idx = self._select_action(game)
                action_tensor = torch.LongTensor([action_idx])
                game_next, reward = TrainerDQN.advance_game(game, action, agent)

                # Observe states
                state_current = torch.from_numpy(game.state())
                state_next = torch.FloatTensor([-1] + [0] * 23) if \
                    game_next.over() else torch.from_numpy(game_next.state())

                # Store the transition in memory
                self._memory.push(state_current,
                                  action_tensor,
                                  state_next,
                                  torch.FloatTensor([reward]))

                game = game_next

                # Perform one step of the optimization (on the target network)
                self._optimize_model()

        aps = actions / (time.time() - time_start)
        print(' :EOF {:.2f} action/second'.format(aps))

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

        player_idx = game.turn_index()
        game_current, _ = game.move(action)
        while game_current.active() and game_current.turn_index() != player_idx:
            if game_current.is_current_player_playing():
                game_current, _ = game_current.move(agent.move(game_current))
            else:
                game_current = game_current.skip_eliminated_player()

        if game_current.over():
            if game_current.winner() == player_idx:
                return game_current, 15
            else:
                return game_current, -5

        return game_current, 0

    def _optimize_model(self):

        if not self._memory.full:
            return
        (state_batch, action_batch, state_next_batch, reward_batch) = \
            self._memory.sample(self._BATCH_SIZE)

        indices_normal = []
        indices_final = []
        for idx, value in enumerate(state_next_batch[:, 0]):
            if value != -1:
                indices_normal.append(idx)
        if len(indices_normal) < 1:
            # escape on randomly empty set
            return

        # Compute a mask of non-final states and concatenate the batch elements
        mask_normal = torch.LongTensor(indices_normal)
        mask_bit_normal = state_next_batch[:, 0] != -1
        if ModelDQN.USE_CUDA:
            mask_bit_normal = mask_bit_normal.cuda()
            # mask_normal = mask_normal.cuda()

        # We don't want to backprop through the expected action values and
        # volatile will save us on temporarily changing the model parameters'
        # requires_grad to False!
        state_batch = ModelDQN.Variable(state_batch)
        action_batch = ModelDQN.Variable(action_batch)
        reward_batch = ModelDQN.Variable(reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self._model(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = ModelDQN.Variable(torch.zeros(self._BATCH_SIZE))
        state_next_batch_normal = ModelDQN.Variable(
            state_next_batch.index_select(0, mask_normal), True)

        next_state_values[mask_bit_normal] = self._model(
            state_next_batch_normal).max(1)[0]

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._GAMMA) + \
            reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        # if self._memory.position % 200 == 0:
        #     print("Loss: ", loss.data[0])

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._model.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    @staticmethod
    def game_to_state(game):
        """Resolve a game's current state to a model input state"""
        return torch.FloatTensor(game._board[0:6] + game._board[7:13]).div(48)


    @staticmethod
    def action_tensor_to_int(action):
        return action[0]

    def _select_action(self, game):
        """
        Based on the current model and a game state, pick a new action
        """
        sample = random.random()
        state = torch.from_numpy(game.state()).type(torch.FloatTensor)
        self._steps_done += 1
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * \
            math.exp(-1. * self._steps_done / self._EPS_DECAY)

        if sample > eps_threshold:
            scores = self._model(
                ModelDQN.Variable(state.unsqueeze(0))
            ).data.cpu().tolist()[0]
            action, _, idx = AgentDQN.action_by_score(
                game, scores, self._steps_done)
        else:
            idx, action = random.choice(AgentDQN.actions_possible(game))

        return action, idx


class ModelDQN(nn.Module):
    """The DQN Model. From the 12 slots, pick of the six choices"""
    USE_CUDA = torch.cuda.is_available()

    def __init__(self):
        super(ModelDQN, self).__init__()
        self.layer1 = nn.Linear(24, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 15)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

    @staticmethod
    def Variable(tensor, volatile=False):
        if ModelDQN.USE_CUDA:
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)

    @staticmethod
    def Seed(seed):
        torch.manual_seed(seed)
        if ModelDQN.USE_CUDA:
            torch.cuda.manual_seed_all(seed)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Stores the transitions that the agent observes, allowing us to reuse this
    data later. By sampling from it randomly, the transitions that build up a
    batch are decorrelated. It has been shown that this greatly stabilizes and
    improves the DQN training procedure.

    http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.states = torch.zeros(capacity, 24)
        self.actions = torch.zeros(capacity, 1).type(torch.LongTensor)
        self.states_next = torch.zeros(capacity, 24)
        self.rewards = torch.zeros(capacity, 1)
        self.full = False
        if ModelDQN.USE_CUDA:
            self.states.cuda()
            self.actions.cuda()
            self.states_next.cuda()
            self.rewards.cuda()

    def push(self, state, action, state_next, reward):
        """Saves a transition."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.states_next[self.position] = state_next
        self.rewards[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size):
        """Sample from the memories"""
        rand_batch = torch.randperm(self.capacity)
        index = torch.split(rand_batch, batch_size)[0]
        states_batch = self.states[index]
        action_batch = self.actions[index]
        states_next_batch = self.states_next[index]
        rewards_batch = self.rewards[index]
        return (states_batch, action_batch, states_next_batch, rewards_batch)

    def __len__(self):
        return self.capacity if self.full else self.position
