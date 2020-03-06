"""
Module with aggregation functions to obtain a solution from a set of cases (and weights)
"""

from collections import defaultdict
import operator

import pandas as pd


class Aggregate:
    """An aggregation procedure to extract solutions from a set of cases"""

    def __init__(self):
        pass

    def aggregate(self, neighbours, similarity):
        """
        Aggregate a set of cases, possibly taking into account their similarities

        Args:
            neighbours (pandas.DataFrame): A dataframe with the most similar cases to aggregate.
            similarity (list of float): A list of the similarities of each case

        Returns:
            The aggregate solution

        """
        raise NotImplementedError

    def get_description(self):
        """Get a dictionary describing the instance"""
        raise NotImplementedError


class MajorityAggregate(Aggregate):
    """Solution aggregation by majority, possible weighted by similarity"""

    def __init__(self, attribute, weighted=True):
        super().__init__()
        self.attribute = attribute
        self.weighted = weighted

        if isinstance(attribute, str) or isinstance(attribute, int):
            self._f = operator.itemgetter(attribute)
        else:
            self._f = attribute

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "attribute": self.attribute, "weighted": self.weighted}

    def aggregate(self, neighbours, similarity):
        d = defaultdict(lambda: 0)
        for (_, n), s in zip(neighbours.iterrows(), similarity):
            d[self._f(n)] += s if self.weighted else 1

        return max(d.items(), key=operator.itemgetter(1))[0]


class ColumnRankAggregate(Aggregate):
    """Solution aggregation by making a ranking of values in a column, possible weighted by similarity"""

    def __init__(self, attributes, true_values=(True,), weighted=True):
        super().__init__()
        self.attributes = attributes
        self.true_values = true_values
        self.weighted = weighted

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "attributes": self.attributes, "true_values": self.true_values,
                "weighted": self.weighted}

    def aggregate(self, neighbours, similarity):
        if not self.weighted:
            n = len(neighbours)
            r = {a: sum(neighbours[a].apply(lambda x: x in self.true_values)) / n for a in self.attributes}
        else:
            s = pd.Series(similarity, index=neighbours.index) / sum(similarity)
            r = {a: sum(neighbours[a].apply(lambda x: x in self.true_values) * s) for a in self.attributes}
        return sorted(list(r.items()), key=operator.itemgetter(1), reverse=True)
