# -*- coding: utf-8 -*-
"""
Love Letter Game object
"""
import numpy as np
from loveletter.card import Card
from loveletter.player import PlayerTools, PlayerAction, PlayerActionTools


class Game():
    """A Love Letter Game"""

    def __init__(self, deck, players, turn_index):
        self._deck = deck
        self._players = players
        self._turn_index = turn_index

        total_playing = sum(
            [1 for player in players if PlayerTools.is_playing(player)])
        self._game_active = total_playing > 1 and self.cards_left() > 0

    def players(self):
        """List of current players."""
        return self._players[:]

    def deck(self):
        """
        List of current cards.

        NOTE: The LAST card [-1] is always held out
        """
        return self._deck

    def draw_card(self):
        """
        Card currently available to the next player.

        Only valid if the game is not over (otherwise No Card)
        """
        return self._deck[0] if len(self._deck) > 1 else Card.noCard

    def held_card(self):
        """
        Card withheld from the game
        """
        return self._deck[-1]

    def turn_index(self):
        """
        Overall turn index of the game.

        This points to the actual action number
        """
        return self._turn_index

    def round(self):
        """Current round number."""
        return self._turn_index // len(self._players)

    def player_turn(self):
        """Player number of current player."""
        return self._turn_index % len(self._players)

    def is_winner(self, idx):
        """True iff that player has won the game"""
        if self.active():
            return False
        player = self._players[idx]
        if not PlayerTools.is_playing(player):
            return False
        other_scores = [
            p.hand_card > player.hand_card for p in self._players if PlayerTools.is_playing(p)]
        return sum(other_scores) == 0

    def winner(self):
        """Return the index of the winning player. -1 if none"""
        for idx in range(len(self._players)):
            if self.is_winner(idx):
                return idx
        return -1

    def player(self):
        """Returns the current player"""
        return self._players[self.player_turn()]

    def opponents(self):
        """Returns the opposing players"""
        return [player for idx, player in enumerate(self._players)
                if idx != self.player_turn() and
                PlayerTools.is_playing(player)]

    def opponent_turn(self):
        """Returns the opposing players indices"""
        return [idx for idx, player in enumerate(self._players)
                if idx != self.player_turn() and
                PlayerTools.is_playing(player)]

    def cards_left(self):
        """
        Number of cards left in deck to distribute

        Does not include the held back card
        """
        return len(self._deck) - 1

    def active(self):
        """Return True if the game is still playing"""
        return self._game_active

    def over(self):
        """Return True if the game is over"""
        return not self.active()

    def is_current_player_playing(self):
        """True if the current player has not been eliminated"""
        return PlayerTools.is_playing(self.player())

    def skip_eliminated_player(self, throw=False):
        """If the current player is eliminated, skip to next"""
        if self.is_current_player_playing():
            return self
        return self._move(PlayerActionTools.blank(), throw)

    def state_hand(self):
        """
        Grab whats in players hand and record it as a one hot encoded array.
        The result is a 16 length binary one hot encoded array
        """
        # whats in hand
        card_number1 = self.player().hand_card
        card_number2 = self.deck()[0]

        cardnumbers = [card_number1, card_number2]
        cardnumbers.sort()
        # initialize arrays
        card1 = np.zeros(8)
        card2 = np.zeros(8)

        # encode whats in hand to array
        card1[cardnumbers[0] - 1] = 1
        card2[cardnumbers[1] - 1] = 1

        return np.concatenate([card1, card2])

    def consumed_cards(self):
        """
        Looks at discarded cards and returns probabilities of outstanding cards.
        """

        cards_discarded = np.array([Game.player_to_discards(
            player) for player in self.players()]).flatten()

        cards_hand = [self.player().hand_card, self.deck()[0]]
        cards_all = np.concatenate([cards_discarded, cards_hand])

        card_bins = np.bincount(cards_all, minlength=9)[1:9]
        card_fractions = card_bins / Card.counts

        return card_fractions

    @staticmethod
    def player_to_discards(player):
        """Returns a list of all cards discarded by player"""
        return [action.discard for action in player.actions]

    def state(self):
        """
        Combines player hand and remaining cards into one array.

        returns numpy float 1d of length 24
        """
        return np.concatenate([self.state_hand(), self.consumed_cards()])

    def _reward(self, game, action):
        """
        Record current reward.
        """
        if game.active():
            if self.is_action_valid(action):
                return 0
            else:
                return -1
        elif game.winner() == self.turn_index():
            return 30
        return -10

    def move(self, action, throw=False):
        """Current player makes an action.

        Returns (NewGame and Reward)<Game,int>
        """
        game = self._move(action)
        return game, self._reward(game, action)

    def _move(self, action, throw=False):
        """Current player makes an action.

        Returns (NewGame and Reward)<Game,int>"""
        if self.over() or not self.is_action_valid(action):
            return self._invalid_input(throw)

        # player is out, increment turn index
        if action.discard == Card.noCard:
            return Game(self.deck(), self.players(), self.turn_index() + 1)

        player = self.player()
        player_hand = [player.hand_card, self._deck[0]]
        player_hand_new = Game.new_hand_card(action.discard, player_hand)
        deck_new = self._deck[1:]

        # choosing to discard the princess ... is valid
        if action.discard == Card.princess:
            return self._move_princess(self._deck[0], deck_new)

        # priest requires modification of action (knowledge)
        if action.discard == Card.priest:
            return self._move_priest(action, player_hand_new, deck_new)

        # updated players for the next turn
        player = PlayerTools.move(self.player(), player_hand_new, action)
        current_players = Game._set_player(
            self._players, player, self.player_turn())

        if action.discard == Card.baron:
            return self._move_baron(action, current_players, player_hand_new, deck_new)

        # No other logic for handmaids or countess
        if action.discard == Card.handmaid or \
                action.discard == Card.countess:
            return Game(deck_new, current_players, self._turn_index + 1)

        if action.discard == Card.guard:
            return self._move_guard(current_players, action, deck_new)

        if action.discard == Card.prince:
            return self._move_prince(current_players, action, deck_new)

        if action.discard == Card.king:
            return self._move_king(current_players, action, deck_new)

        raise NotImplementedError("Missing game logic")

    def _move_guard(self, current_players, action, deck_new):
        """
        Handle a guard action into a new game state

        Player makes a guess to try and eliminate the opponent
        """
        if self._players[action.player_target].hand_card == action.guess and \
                not PlayerTools.is_defended(self._players[action.player_target]):
            # then target player is out
            player_target = PlayerTools.force_discard(
                self._players[action.player_target])
            current_players = Game._set_player(
                current_players, player_target, action.player_target)

        return Game(deck_new, current_players, self._turn_index + 1)

    def _move_priest(self, action, player_hand_new, deck_new):
        """
        Handle a priest action into a new game state

        Action gains knowledge of other player's card
        """
        player_targets_card = Card.noCard if \
            PlayerTools.is_defended(self._players[action.player_target]) \
            else self._players[action.player_target].hand_card
        action_updated = PlayerAction(
            action.discard, action.player_target, action.guess, player_targets_card)

        player = PlayerTools.move(
            self.player(), player_hand_new, action_updated)
        current_players = Game._set_player(
            self._players, player, self.player_turn())

        return Game(deck_new, current_players, self._turn_index + 1)

    def _move_baron(self, action, current_players, player_hand_new, deck_new):
        """
        Handle a baron action into a new game state

        Player and target compare hand cards. Player with lower hand
        card is eliminated
        """
        card_target = self._players[action.player_target].hand_card
        if player_hand_new > card_target:
            if not PlayerTools.is_defended(self._players[action.player_target]):
                # target is eliminated
                player_target = PlayerTools.force_discard(
                    self._players[action.player_target])
                current_players = Game._set_player(
                    current_players, player_target, action.player_target)
        else:
            # player is eliminated
            player = PlayerTools.force_discard(self.player(), player_hand_new)
            player = PlayerTools.force_discard(player)
            current_players = Game._set_player(
                current_players, player, self.player_turn())

        return Game(deck_new, current_players, self._turn_index + 1)

    def _move_prince(self, current_players, action, deck_new):
        """Handle a prince action into a new game state"""

        player_before_discard = current_players[action.player_target]

        # if there are no more cards, this has no effect
        if len(deck_new) - 1 < 1:
            return Game(deck_new, current_players, self._turn_index + 1)

        if player_before_discard.hand_card == Card.princess:
            player_post_discard = PlayerTools.force_discard(
                player_before_discard)
            deck_final = deck_new
        else:
            player_post_discard = PlayerTools.force_discard(
                player_before_discard, deck_new[0])
            deck_final = deck_new[1:]

        current_players = Game._set_player(
            current_players, player_post_discard, action.player_target)

        return Game(deck_final, current_players, self._turn_index + 1)

    def _move_king(self, current_players, action, deck_new):
        """Handle a king action into a new game state"""
        player = current_players[self.player_turn()]
        target = current_players[action.player_target]

        player_new = PlayerTools.set_hand(player, target.hand_card)
        target_new = PlayerTools.set_hand(target, player.hand_card)

        current_players = Game._set_player(
            current_players, player_new, self.player_turn())
        current_players = Game._set_player(
            current_players, target_new, action.player_target)

        return Game(deck_new, current_players, self._turn_index + 1)

    def _move_princess(self, dealt_card, new_deck):
        """Handle a princess action into a new game state"""
        player = PlayerTools.force_discard(self.player(), dealt_card)
        player = PlayerTools.force_discard(player)
        current_players = Game._set_player(
            self._players, player, self.player_turn())
        return Game(new_deck, current_players, self._turn_index + 1)

    def is_action_valid(self, action):
        """Tests if an action is valid given the current game state"""
        player = self.player()

        # if player is out, only valid action is no action
        if player.hand_card == Card.noCard:
            return PlayerActionTools.is_blank(action)

        # cannot target an invalid player
        if not self._is_valid_player_target(action.player_target):
            return False

        target_player = self._players[action.player_target]
        player_hand = [player.hand_card, self._deck[0]]

        # cannot discard a card not in the hand
        if action.discard not in player_hand:
            return False

        new_hand_card = Game.new_hand_card(action.discard, player_hand)

        # countess must be discarded if the other card is king/prince
        if new_hand_card == Card.countess and \
                (action.discard == Card.prince or action.discard == Card.king):
            return False

        # cannot mis-target a card
        if self.player_turn() == action.player_target and action.discard in Card.only_other:
            return False
        if self.player_turn() != action.player_target and action.discard in Card.only_self:
            return False

        if not PlayerTools.is_playing(target_player):
            return False

        # Cannot guess guard or no card
        if action.discard == Card.guard and (
                action.guess == Card.guard or action.guess == Card.noCard):
            return False

        return True

    def _is_valid_player_target(self, player_target):
        """True iff the player can be targeted by an action"""
        if player_target < 0 or player_target >= len(self._players):
            return False

        return PlayerTools.is_playing(self._players[player_target])

    def _invalid_input(self, throw):
        """Throw if true, otherwise return current game"""
        if throw:
            raise Exception("Invalid Move")
        return self

    def to_str(self):
        """Returns a string[] representation of the game"""
        strings = [
            "" + ("━" * 79),
            "Game is active" if self.active() else "Game is over",
            "Round:{: >2} | Cards Left:{: >2} | Withheld Card: {: >10} ".format(
                self.round(), self.cards_left(), Card.render_card_number(self.held_card())),
            ""
        ]
        for idx, player in enumerate(self._players):
            strings += self._to_str_player(idx, player)
            strings += [""]

        return strings

    def _to_str_player(self, idx, player):
        is_playing = " " if PlayerTools.is_playing(player) else "☠️"
        is_turn = "⭐" if self.player_turn() == idx else " "
        draw_card = self.draw_card() if self.active(
        ) and self.player_turn() == idx else Card.noCard
        draw_card_render = Card.render_card_number(draw_card)
        header = "Player {} {} {}".format(idx, is_turn, is_playing)
        state = "   Current: {} {}".format(
            draw_card_render, PlayerTools.to_str(player))
        return [header, state]

    @staticmethod
    def _set_player(players, player_new, player_new_index):
        """Return a fresh copy of players with the new player in the index"""
        players_new = players[:]
        players_new[player_new_index] = player_new
        return players_new

    @staticmethod
    def new_hand_card(card_discard, hand):
        """New hand card based on current cards in hand"""
        new_hand = list(filter(lambda card: card != card_discard, hand))
        if len(new_hand) < 1:
            # this means the hand contained only one card. so one still remains
            return card_discard
        return int(new_hand[0])

    @staticmethod
    def new(player_count=4, seed=451):
        """Create a brand new game"""
        deck = Card.shuffle_deck(seed)

        dealt_cards = deck[:player_count]
        undealt_cards = deck[player_count:]

        players = list(map(PlayerTools.blank, dealt_cards))
        return Game(undealt_cards, players, 0)
