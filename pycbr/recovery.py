import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer


def _nan_average(a, weights):
    """NaN-compatible weighted average"""
    masked_data = np.ma.masked_array(a, np.isnan(a))
    return np.ma.average(masked_data, weights=weights)


class _WeightedDistance:
    """A distance function calculated from a weighted average of similarities"""

    def __init__(self, similarities, weights=None):
        self.similarities = similarities
        self.weights = weights

    def __call__(self, x, y):
        sims = [s(a, b) for s, a, b in zip(self.similarities, x, y)]
        if self.weights is None:
            return 1 - np.nanmean(sims)
        return 1 - _nan_average(sims, self.weights)


class Recovery:
    """A case recovery system"""

    def __init__(self, attributes, nafill="n.a."):
        """

        Args:
            attributes (list of tuples): A list of 3-tuples with attribute name, Attribute instance defining its
                                         similarity and a weight. A list of 2-tuples can also be given if weight is
                                         uniform.
            nafill: A value used to replace na. Should be compatible with the Attribute instances.
        """
        self.attributes = attributes
        self.nafill = nafill

        if any(len(x) == 3 for x in attributes):  # At least one weight
            # An error may raise if a weight is absent
            if any(len(x) != 3 for x in attributes):
                raise ValueError("Inconsistent attribute format.")
            self.weights = [a[2] for a in attributes]
        else:
            self.weights = None

        self.transformer = ColumnTransformer([(a[0], a[1], [a[0]]) for a in attributes])

        self.distance = _WeightedDistance([a[1].similarity for a in attributes], weights=self.weights)

        self.searcher = NearestNeighbors(metric=self.distance)

        self.df = None

    def fit(self, X):
        """
        Prepare the Recovery system with a case base.

        Args:
            X (pd.DataFrame): A dataframe with the cases.

        """
        self.searcher.fit(self.transformer.fit_transform(X.fillna(self.nafill)))
        self.df = X

    def find(self, X, k):
        """
        Get the most similar cases to a set of target new cases.

        Args:
            X (pd.DataFrame): A dataframe with the new cases.
            k (int): Amount of most-similar cases.

        Returns:
            list of (pandas.DataFrame, np.array of float): List of dataframes with the most similar cases and
                                                           similarity scores.
        """
        distances, neigh = self.searcher.kneighbors(self.transformer.transform(X.fillna(self.nafill)), k)

        return [(self.df.iloc[n], 1 - d) for n, d in zip(neigh, distances)]
