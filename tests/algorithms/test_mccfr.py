import numpy as np
import pytest

from gamegym.games.matrix import MatchingPennies, RockPaperScissors, ZeroSumMatrixGame
from gamegym.games.goofspiel import Goofspiel
from gamegym.algorithms.bestresponse import BestResponse
from gamegym.algorithms.mccfr import OutcomeMCCFR

def test_regret():
    g = MatchingPennies()
    mc = OutcomeMCCFR(g, seed=42)
    rs = mc.regret_matching(np.array([-1.0, 0.0, 1.0, 2.0]))
    assert rs == pytest.approx([0.0, 0.0, 1.0 / 3, 2.0 / 3])

def test_pennies():
    np.set_printoptions(precision=3)
    g = MatchingPennies()
    g = RockPaperScissors()
    g = ZeroSumMatrixGame([[1, 0], [0, 1]])
    mc = OutcomeMCCFR(g, seed=12)
    mc.compute(1000)
    s = g.initial_state()
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1
    s = s.play("H")
    assert np.max(np.abs(mc.distribution(s).probabilities() - [0.5, 0.5])) < 0.1

def test_mccfr_goofspiel():
    g = Goofspiel(3)
    mc = OutcomeMCCFR(g, seed=56)
    mc.compute(1000)
    br = BestResponse(g, 0, {1:mc})
    assert np.mean([
        g.play_strategies([br, mc], seed=i)[-1].values()[0]
        for i in range(1000)]) == pytest.approx(0.0, abs=0.2)
