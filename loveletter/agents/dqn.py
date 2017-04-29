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

            while game.active() and game.is_current_player_playing():
                # Select and perform an action
                actions += 1
                action, action_idx = self._select_action(game)
                action_tensor = torch.LongTensor([action_idx])
                game_next, reward = TrainerDQN.advance_game(game, action, agent)

                # Observe states
                state_current = torch.from_numpy(game.state())
                state_next = None if game_next.over() else torch.from_numpy(game_next.state())
                # print(game_next.over())
                # print(state_next)

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

        player_idx = game.player_turn()
        game_current, _ = game.move(action)
        while game_current.active() and game_current.player_turn() != player_idx:
            if game_current.is_current_player_playing():
                game_current, _ = game_current.move(agent.move(game_current))
            else:
                game_current = game_current.skip_eliminated_player()

        # print("Round", game.round(), '->', game_current.round())

        if game_current.over():
            if game_current.winner() == player_idx:
                return game_current, 15
            else:
                return game_current, -5

        return game_current, 0

    def _optimize_model(self):

        if len(self._memory) < self._BATCH_SIZE:
            return
        transitions = self._memory.sample(self._BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))
        if ModelDQN.USE_CUDA:
            non_final_mask = non_final_mask.cuda()
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        if len(batch.next_state) < 1:
            non_final_next_states = ModelDQN.Variable(torch.cat([s for s in batch.next_state
                                                        if s is not None]),
                                                        volatile=True)
            print("NONSKIP")
        else:
            print("SKIP")
            return
        state_batch = ModelDQN.Variable(torch.cat(batch.state))
        action_batch = ModelDQN.Variable(torch.cat(batch.action))
        reward_batch = ModelDQN.Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self._BATCH_SIZE))
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


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
        self.layer1 = nn.Linear(24, 2048)
        self.layer2 = nn.Linear(2048, 4096)
        self.layer3 = nn.Linear(4096, 2048)
        self.layer4 = nn.Linear(2048, 15)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.dropout(self.layer1(x)))
        x = F.relu(self.dropout(self.layer2(x)))
        x = F.relu(self.dropout(self.layer3(x)))
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
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
