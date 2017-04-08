# -*- coding: utf-8 -*-
"""Simple, face up console version of game"""

import argparse

import numpy as np

from loveletter.game import Game
from loveletter.player import PlayerAction, PlayerActionTools
from loveletter.card import Card


PARSER = argparse.ArgumentParser(
    description='Play a console Love Letter game')

PARSER.add_argument('--seed', type=int, default=451,
                    help='Seed to populate game')
PARSER.add_argument('--replay', type=str, default="",
                    help='Actions to replay (copied from previous output)')

ARGS = PARSER.parse_args()

# def actions_to_str(actions):
#     np_arr = [PlayerActionTools.to_np_many]


def display(game, actions):
    """Print a game to the console"""
    lst = [str(i) for i in PlayerActionTools.to_np_many(actions)]
    print(",".join(lst))
    for line in game.to_str():
        print(line)


def get_int(prompt):
    """Prompt until proper value given"""
    print(prompt)
    while True:
        try:
            return int(input(" > "))
        except ValueError:
            print("Invalid Entry - Exit with Ctrl-C")


def get_action():
    """Get a player action from the console"""
    discard = get_int("Discard Card")
    player_target = get_int("Player Target")
    guess = get_int("Guessed Card") if discard == Card.guard else 0
    return PlayerAction(discard, player_target, guess, 0)


def play(seed, previous_actions):
    """Play a game"""
    game = Game.new(4, seed)
    previous_actions = np.array([], dtype=np.uint8) if len(previous_actions) < 1 else \
        np.array([int(i) for i in previous_actions.split(",")], dtype=np.uint8)
    previous_actions = PlayerActionTools.from_np_many(previous_actions)[::-1]
    actions = []
    while game.active():
        if not game.is_current_player_playing():
            game = game.skip_eliminated_player()
            continue

        display(game, actions)

        try:
            if len(previous_actions) > 0:
                action = previous_actions.pop()
            else:
                print("  What card to play?")
                action = get_action()
            actions.append(action)
            game = game.move(action)
        except ValueError:
            print("Invalid move - Exit with Ctrl-C")

    print("Game Over")

if __name__ == "__main__":
    play(ARGS.seed, ARGS.replay)
