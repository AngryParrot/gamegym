from ..game import Game, GameState
from ..distribution import Uniform

class Goofspiel(Game):

    def __init__(self, n_cards):
        self.cards = tuple(range(1, n_cards + 1))

    def initial_state(self):
        return GoofspielState(None, None, game=self)

    def players(self):
        return 2


def determine_winner(card1, card2):
    if card1 > card2:
        return 1
    if card2 > card1:
        return 2
    return 0


class GoofspielState(GameState):

    def player(self):
        if len(self.history) == len(self.game.cards) * 3:
            return -1
        return len(self.history) % 3

    def round(self):
        return len(self.history) // 3

    def cards_in_hand(self, player):
        cards = list(self.game.cards)
        for c in self.played_cards(player):
            cards.remove(c)
        return cards

    def played_cards(self, player):
        return self.history[player::3]

    def actions(self):
        return self.cards_in_hand(self.player())

    def chance_distribution(self):
        return Uniform(self.actions())

    def winners(self):
        cards1 = self.played_cards(1)
        cards2 = self.played_cards(2)
        return [determine_winner(c1, c2) for c1, c2 in zip(cards1, cards2)]

    def score(self, player):
        return sum(v for w, v in zip(self.winners(), self.played_cards(0))
                   if w == player)

    def values(self):
        s1 = self.score(1)
        s2 = self.score(2)
        if s1 < s2:
            return (-1, 1)
        elif s1 > s2:
            return (1, -1)
        else:
            return (0, 0)

    def player_information(self, player):
        return (len(self.history),
                self.winners(),
                self.played_cards(0),
                self.played_cards(player))


def test_goofspeil():
    import pytest
    g = Goofspiel(7)
    s = g.initial_state()

    assert s.player() == 0
    assert s.score(1) == 0
    assert s.score(2) == 0
    assert s.actions() == list(range(1, 8))
    assert s.chance_distribution().probabilities() == (pytest.approx(1 / 7),) * 7

    for i, a in enumerate(
                [4, 2, 1,
                 5, 4, 6,
                 6, 3, 3,
                 2, 5, 4,
                 3, 7]):
        s = s.play(a)
        s.player() == i % 3

    assert s.round() == 4
    assert s.player() == 2
    assert s.actions() == [2, 5, 7]
    assert s.winners() == [1, 2, 0, 1]
    assert s.chance_distribution().probabilities() == (pytest.approx(1/3),) * 3
    assert s.score(1) == 6
    assert s.score(2) == 5

    assert s.cards_in_hand(0) == [1, 7]
    assert s.cards_in_hand(1) == [1, 6]
    assert s.cards_in_hand(2) == [2, 5, 7]

    for a in [2,
              7, 6, 7,
              1, 1, 1]:
        s = s.play(a)

    assert s.is_terminal()
    assert s.score(1) == 9
    assert s.score(2) == 12

    assert s.values() == (-1, 1)