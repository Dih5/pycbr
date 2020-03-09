"""
Module with models to define attribute similarity
"""

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn import base


class Attribute(base.TransformerMixin, base.BaseEstimator):
    """Generic attribute class. Defines how an attribute is transformed and how the similarity is calculated"""

    def __init__(self):
        pass

    def similarity(self, x, y):
        raise NotImplementedError

    def get_description(self):
        """Get a dictionary describing the instance"""
        raise NotImplementedError


class LinearAttribute(Attribute):
    """A continuous attribute whose similarity is measured with a linear function

    The attribute similarity function is defined by
    :math:`\\mathrm{sim}_a(x, y)= \\max\\left(1-\\frac{\\left|x-y\\right|}{a},0\\right)`

    """

    def __init__(self, max_value=None):
        super().__init__()
        self.max_value = max_value

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "max_value": self.max_value}

    def fit(self, X, y=None):
        if self.max_value is None:
            raise NotImplementedError("Automatic range not yet available")
        return self

    def transform(self, X, y=None):
        return X

    def similarity(self, x, y):
        return max(1 - abs(x - y) / self.max_value, 0)


class ExponentialAttribute(Attribute):
    """A continuous attribute whose similarity is measured with a exponential function

    The attribute similarity function is defined by
    :math:`\\mathrm{sim}_a(x, y)= a^{\\left|x-y\\right|}`,
    where :math:`a \\in [0, 1]`.

    """

    def __init__(self, base=None):
        super().__init__()
        assert 0 <= base <= 1
        self.base = base

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "base": self.base}

    def fit(self, X, y=None):
        if self.base is None:
            raise NotImplementedError("Automatic range not yet available")
        return self

    def transform(self, X, y=None):
        return X

    def similarity(self, x, y):
        return self.base ** abs(x - y)


class QuantileLinearAttribute(Attribute):
    """A continuous attribute whose similarity is measured with a linear function on the quantiles

    The attribute similarity function is defined by
    :math:`\\mathrm{sim}(x, y)= 1-\\left|Q_A(x)-Q_A(y)\\right|`, where :math:`Q_A` is the (sample) quantile function
    for the attribute in consideration.

    """

    def __init__(self):
        super().__init__()
        self.encoder = None

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__, }

    def fit(self, X, y=None):
        self.encoder = QuantileTransformer()
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def similarity(self, x, y):
        return 1 - abs(x - y)


class KroneckerAttribute(Attribute):
    """A (possibly) categorical attribute whose similarity is 1 if equal or 0 otherwise

    The attribute similarity function is defined by
    :math:`\\mathrm{sim}(x, y)= \\delta_{x,y}`

    """

    def __init__(self, encode=True, undefined=("n.a.",)):
        super().__init__()
        self.encode = encode
        self.undefined = undefined

        self.encoder = None
        self.encoded_undefined = []

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "encode": self.encode, "undefined": self.undefined}

    def fit(self, X, y=None):
        if self.encode:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(np.concatenate((X, np.asarray([list(self.undefined)]))))
            self.encoded_undefined = self.encoder.transform([list(self.undefined)]).tolist()
        else:
            self.encoded_undefined = self.undefined
        return self

    def transform(self, X, y=None):
        if self.encode:
            return self.encoder.transform(X)
        return X

    def similarity(self, x, y):
        if (x in self.encoded_undefined) or (y in self.encoded_undefined):
            return np.nan
        return 1 if x == y else 0


class LinearOrdinalAttribute(Attribute):
    """A (possibly) categorical attribute whose similarity is linear with respect to a scale"""

    def __init__(self, order, undefined=("n.a.",)):
        """

        Args:
            order (list): List of values, defining their ordering.
            undefined (iterable): Values which are recognized, but not comparable to the ranking. When such a value is
                                  found, the similarity returned is nan.
        """
        super().__init__()
        self.order = order
        self.undefined = undefined

        self.n = len(order)
        self.encoder = None

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "order": self.order, "undefined": self.undefined}

    def fit(self, X, y=None):
        self.encoder = OrdinalEncoder([self.order + list(self.undefined)])
        self.encoder.fit([[x] for x in self.order + list(self.undefined)])  # Argument irrelevant
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def similarity(self, x, y):
        if x >= self.n or y >= self.n:
            return np.nan
        return 1 - abs(x - y) / (self.n - 1)


class MatrixOrdinalAttribute(Attribute):
    """A (possibly) categorical attribute whose similarity is defined by a matrix"""

    def __init__(self, values, matrix, undefined=("n.a.",)):
        super().__init__()
        self.values = values
        self.matrix = matrix
        self.undefined = undefined

        self.n = len(values)
        self.encoder = None

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "values": self.values, "matrix": self.matrix,
                "undefined": self.undefined}

    def fit(self, X, y=None):
        self.encoder = OrdinalEncoder([self.values + list(self.undefined)], dtype=int)
        self.encoder.fit([[x] for x in self.values + list(self.undefined)])  # Argument irrelevant
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def similarity(self, x, y):
        if x >= self.n or y >= self.n:
            return np.nan
        return self.matrix[x][y]
