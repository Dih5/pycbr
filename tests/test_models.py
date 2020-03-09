import numpy as np

from pycbr import models

eps = 1E-6


def test_load():
    """Trivial test to if package loads"""
    assert True


def test_matrix_similarity():
    """Test matrix-defined similarities"""
    values = list("ABCD")
    a = models.MatrixOrdinalAttribute(values, [[1.0, 0.5, 0.2, 0.0],
                                               [0.5, 1.0, 0.5, 0.2],
                                               [0.2, 0.5, 1.0, 0.5],
                                               [0.0, 0.2, 0.5, 1.0]])

    a.fit(None)

    for x, y, s in [("A", "A", 1.0),
                    ("A", "B", 0.5),
                    ("D", "A", 0.0),
                    ("D", "D", 1.0)]:
        x2 = a.transform([[x]])[0, 0]
        y2 = a.transform([[y]])[0, 0]
        assert a.similarity(x2, y2) == s

    y2 = a.transform([["n.a."]])[0, 0]
    assert np.isnan(a.similarity(x2, y2))


def test_linear_rank_similarity():
    """Test ordinal rank-defined similarities"""
    values = list("ABCD")
    a = models.LinearOrdinalAttribute(values)

    a.fit(None)

    for x, y, s in [("A", "A", 1.0),
                    ("A", "B", 2 / 3),
                    ("A", "C", 1 / 3),
                    ("B", "C", 2 / 3),
                    ("D", "D", 1.0)]:
        x2 = a.transform([[x]])[0, 0]
        y2 = a.transform([[y]])[0, 0]
        assert abs(a.similarity(x2, y2) - s) < eps

    y2 = a.transform([["n.a."]])[0, 0]
    assert np.isnan(a.similarity(x2, y2))


def test_linear_continuous_similarity():
    """Test linear similarities"""
    a = models.LinearAttribute(100)

    a.fit(None)

    for x, y, s in [(10, 10, 1.0),
                    (20, 20, 1.0),
                    (0, 100, 0.0),
                    (50, 0, 1 - 50 / 100),
                    (20, 40, 1 - 20 / 100)]:
        assert abs(a.similarity(x, y) - s) < eps


def test_exponential_continuous_similarity():
    """Test exponential similarities"""
    a = models.ExponentialAttribute(0.5)

    a.fit(None)

    for x, y, s in [(10, 10, 1.0),
                    (20, 20, 1.0),
                    (0, 3, 0.5 ** 3),
                    (3, 0, 0.5 ** 3),
                    ]:
        assert abs(a.similarity(x, y) - s) < eps
