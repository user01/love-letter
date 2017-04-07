# -*- coding: utf-8 -*-
"""Simple, face up console version of game"""

from loveletter.game import Game
from loveletter.player import PlayerAction
from loveletter.card import Card

def display(game):
    """Print a game to the console"""
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

def play():
    """Play a game"""
    game = Game.new()
    while game.active():
        display(game)
        print("  What card to play?")

        try:
            action = get_action()
            game = game.move(action)
        except ValueError:
            print("Invalid move - Exit with Ctrl-C")




if __name__ == "__main__":
    play()
