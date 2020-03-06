"""
Module providing the functionality to build a recovery system
"""

import numpy as np
import pandas as pd
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

    def __init__(self, attributes, na_strategy="drop", na_fill="n.a.", algorithm="auto"):
        """

        Args:
            attributes (list of tuples): A list of 3-tuples with attribute name, Attribute instance defining its
                                         similarity and a weight. A list of 2-tuples can also be given if weight is
                                         uniform.
            na_strategy (str): Method to deal with not available data. Possible options are:
                               - "drop": Ignore the cases with not available values (among those defining the
                                         similarity)
                               - "replace": Replace na with a special value provided by the na_fill parameter.
                                         Similarity functions must be compatible with it.
            na_fill: A value used to replace na. Should be compatible with the Attribute instances.
            algorithm (str): Method to retrieve the nearest neighbours. Available options are "auto", "ball_tree",
                             "kd_tree", and "brute". If the attribute parameter is not actually defining a metric
                             (e.g., non-transitive) the "brute" method must be used.
        """
        self.attributes = attributes
        self.na_strategy = na_strategy.lower()
        self.na_fill = na_fill
        self.algorithm = algorithm

        if any(len(x) == 3 for x in attributes):  # At least one weight
            # An error may raise if a weight is absent
            if any(len(x) != 3 for x in attributes):
                raise ValueError("Inconsistent attribute format.")
            self.weights = [a[2] for a in attributes]
        else:
            self.weights = None

        self.transformer = ColumnTransformer([(a[0], a[1], [a[0]]) for a in attributes])

        self.distance = _WeightedDistance([a[1].similarity for a in attributes], weights=self.weights)

        self.searcher = NearestNeighbors(metric=self.distance, algorithm=algorithm)

        # Training df
        self.df = None
        # Transformed training df (possibly dropping instances)
        self.transformed = None
        # Index of the transformed instances in the original df
        self._index = None

    def get_description(self):
        attributes = [[a[0]] + [a[1].get_description()] + list(a[2:]) for a in self.attributes]
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "attributes": attributes, "na_strategy": self.na_strategy,
                "na_fill": self.na_fill, "algorithm": self.algorithm}

    def _deal_with_na(self, X):
        """Transform a dataframe according to the na_strategy of the instance"""
        if self.na_strategy == "drop":
            return X.dropna(subset=[x[0] for x in self.attributes])
        elif self.na_strategy == "replace":
            return X.fillna(self.na_fill)
        else:
            raise ValueError("Invalid na_strategy: %s" % self.na_strategy)

    def fit(self, X):
        """
        Prepare the Recovery system with a case base.

        Args:
            X (pd.DataFrame): A dataframe with the cases.

        """
        # Store the original CB
        self.df = X

        # Imputate/drop
        X2 = self._deal_with_na(X[[x[0] for x in self.attributes]])

        # Save the index for later
        self._index = X2.index.copy()

        # Transform according to similarities (numpy array)
        self.transformed = self.transformer.fit_transform(X2)

        # Fit the neighbour search
        self.searcher.fit(self.transformed)

        # Store the transformed CB as a dataframe
        self.transformed = pd.DataFrame(self.transformed, index=self._index, columns=[x[0] for x in self.attributes])

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
        distances, neigh = self.searcher.kneighbors(self.transformer.transform(X[[x[0] for x in self.attributes]]), k)

        # Note NearestNeighbors returns indices from its input. Hence, iloc and not loc must be used in the dataframe
        return [(self.df.loc[self._index[n]], 1 - d) for n, d in zip(neigh, distances)]
