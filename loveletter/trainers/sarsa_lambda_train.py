'''
The idea here is to have two agents, both using SARSA Lambda
playing the game 1 on 1. After each game, we average their
action values.
'''

import numpy as np
from loveletter.game import Game
import random
from collections import namedtuple
import os.path

num_iterations = 5000
num_players = 2
epsilon = [0.1, 0.2]
alpha = 0.5
gamma = 1
lam = 0.5

# neural network state space to traditional state space
def nn_to_ss(state):
    lh = np.argmax(state[0:8])
    rh = np.argmax(state[8:16])
    cards = np.multiply(state[16:24], [5, 2, 2, 2, 2, 1, 1, 1]).astype(int)

    return np.concatenate(([lh, rh], cards))

def action_idx_to_action(idx, t):
    player = t
    opponent = 1 - t

    PlayerAction = namedtuple(
    'PlayerAction', 'discard player_target guess revealed_card')

    if idx == 0:
        return PlayerAction(discard = 1, player_target = opponent, guess = 2, revealed_card = 0)
    if idx == 1:
        return PlayerAction(discard = 1, player_target = opponent, guess = 3, revealed_card = 0)
    if idx == 2:
        return PlayerAction(discard = 1, player_target = opponent, guess = 4, revealed_card = 0)
    if idx == 3:
        return PlayerAction(discard = 1, player_target = opponent, guess = 5, revealed_card = 0)
    if idx == 4:
        return PlayerAction(discard = 1, player_target = opponent, guess = 6, revealed_card = 0)
    if idx == 5:
        return PlayerAction(discard = 1, player_target = opponent, guess = 7, revealed_card = 0)
    if idx == 6:
        return PlayerAction(discard = 1, player_target = opponent, guess = 8, revealed_card = 0)
    if idx == 7:
        return PlayerAction(discard = 2, player_target = opponent, guess = 0, revealed_card = 0)
    if idx == 8:
        return PlayerAction(discard = 3, player_target = opponent, guess = 0, revealed_card = 0)
    if idx == 9:
        return PlayerAction(discard = 4, player_target = player, guess = 0, revealed_card = 0)
    if idx == 10:
        return PlayerAction(discard = 5, player_target = player, guess = 0, revealed_card = 0)
    if idx == 11:
        return PlayerAction(discard = 5, player_target = opponent, guess = 0, revealed_card = 0)
    if idx == 12:
        return PlayerAction(discard = 6, player_target = opponent, guess = 0, revealed_card = 0)
    if idx == 13:
        return PlayerAction(discard = 7, player_target = player, guess = 0, revealed_card = 0)
    if idx == 14:
        return PlayerAction(discard = 8, player_target = player, guess = 0, revealed_card = 0)



# 2nd for each available action, of which there are 15
# 1st for which player, 0 for player 0, 1 for player 1
# 3rd for left hand card, 3rd for right hand card
# rest of numbers show how much 
Q = np.random.rand(num_players, 15, 8, 8, 6, 3, 3, 3, 3, 2, 2, 2)

if os.path.exists("loveletter/models/sarsa_lambda_Q.npz"):
    with np.load('loveletter/models/sarsa_lambda_Q.npz') as data:
        Q[0] = data['Q']
        Q[1] = data['Q']

wins = np.zeros(num_players).astype(int)

for i in range(num_iterations):

    has_moved = [False] * num_players
    R = np.zeros(num_players)
    A = np.zeros(num_players).astype(int)
    Ap = np.zeros(num_players).astype(int)
    S = np.zeros((2, len(Q.shape) - 2)).astype(int)
    Sp = np.zeros((2, len(Q.shape) - 2)).astype(int)

    game = Game.new(player_count=2, seed=random.seed())

    t = 0
    first_round = True

    E = np.zeros(Q.shape)

    while(game.active()):
        if not has_moved[t]:
            S[t] = nn_to_ss(game.state())

            if (np.random.random() > epsilon[t]):
                A[t] = np.argmax(Q[(t, ...) + tuple(S[t])])
            else:
                A[t] = np.random.randint(15)

        game, R[t] = game.move(action_idx_to_action(A[t], t))
        has_moved[t] = True

        # get the current players turn....
        t = game.player_turn()

        if has_moved[t] and game.active():
            Sp[t] = nn_to_ss(game.state())

            if (np.random.random() > epsilon[t]):
                Ap[t] = np.argmax(Q[(t, ...) + tuple(S[t])])
            else:
                Ap[t] = np.random.randint(15)

            delta = R[t] + gamma * Q[(t, Ap[t]) + tuple(Sp[t])] - Q[(t, A[t]) + tuple(S[t])]

            E[(t, A[t]) + tuple(S[t])] = E[(t, A[t]) + tuple(S[t])] + 1

            Q[t] = Q[t] + (alpha * delta * E[t])
            E[t] = gamma * lam * E[t]

            S[t] = Sp[t]
            A[t] = Ap[t]

    # If the game has ended.... set rewards for everyone who has moved
    for p in range(num_players):
        if has_moved[p]:
            delta = R[p] + gamma * Q[(p, Ap[p]) + tuple(Sp[p])] - Q[(p, A[p]) + tuple(S[p])]

            E[(p, A[p]) + tuple(S[p])] = E[(p, A[p]) + tuple(S[p])] + 1

            Q[p] = Q[p] + (alpha * delta * E[p])
            E[p] = gamma * lam * E[p]
    
    print("Game Over. Player " + str(game.winner()) + " won.")
    wins[game.winner()] += 1

    #this is hard-coded because honestly we just gotta move at this point
    Q[0] = np.divide(np.add(Q[0], Q[1]), 2)
    Q[1] = Q[0]

np.savez_compressed("loveletter/models/sarsa_lambda_Q.npz", Q=Q[0])
print("Games over. Q saved.")
