import numpy as np
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn import base


class Attribute(base.TransformerMixin, base.BaseEstimator):
    """Attribute class. Defines how an attribute is transformed and how the similarity is calculated"""

    def __init__(self):
        pass

    def similarity(self, x, y):
        raise NotImplementedError


class LinearAttribute(Attribute):
    """A continuous attribute whose similarity is measured with a linear function"""

    def __init__(self, max_value=None):
        super().__init__()
        self.max_value = max_value

    def fit(self, X, y=None):
        if self.max_value is None:
            raise NotImplementedError("Automatic range not yet available")
        return self

    def transform(self, X, y=None):
        return X

    def similarity(self, x, y):
        return max(1 - abs(x - y) / self.max_value, 0)


class QuantileLinearAttribute(Attribute):
    """A continuous attribute whose similarity is measured with a linear function on the quantiles"""

    def __init__(self):
        super().__init__()
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = QuantileTransformer()
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def similarity(self, x, y):
        return 1 - abs(x - y)


class KroneckerAttribute(Attribute):
    """A (possibly) categorical attribute whose similarity is 1 if equal or 0 otherwise"""

    def __init__(self, encode=True, undefined=("n.a.",)):
        super().__init__()
        self.encode = encode
        self.undefined = undefined

        self.encoder = None
        self.encoded_undefined = []

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
    """A (possibly) categorical attribute whose similarity is linear wrt a scale"""

    def __init__(self, order, undefined=("n.a.",)):
        super().__init__()
        self.order = order
        self.undefined = undefined

        self.n = len(order)
        self.encoder = None

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
