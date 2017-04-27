# -*- coding: utf-8 -*-
"""Find games from seed with certain cards"""

import argparse

from loveletter.game import Game

PARSER = argparse.ArgumentParser(
    description='Find a game seed with desired card set ')

PARSER.add_argument('--cards', type=str,
                    help='Cards to find in comma delimited list')

ARGS = PARSER.parse_args()

def find_seed(cards_str):
    """Play a game"""
    cards = set([int(s) for s in cards_str.split(',')])
    results = 0
    for idx in range(5000):
        game = Game.new(4, idx)
        # available_cards = set([player.hand_card for player in game.players()] + [game.draw_card()])
        available_cards = set(
            [game.players()[0].hand_card] + [game.draw_card()])
        if cards.issubset(available_cards):
            print("Found at {: >5}".format(idx))
            results += 1
        if results > 5:
            return




if __name__ == "__main__":
    find_seed(ARGS.cards)
