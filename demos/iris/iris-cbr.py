#!/usr/bin/env python3
"""Minimal CBR demo with the Iris Dataset"""
import pycbr

import pandas as pd
from sklearn import datasets
import tempfile

# Load the Iris dataset to build the example
iris = datasets.load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["species"] = iris["target"]
df["species"] = df["species"].apply(lambda x: iris["target_names"][x])

# Store it in a temporal file


f = tempfile.NamedTemporaryFile(suffix=".csv")
df.to_csv(f.name, index=False)

# The steps to build a CBR are following:
# 1- Define a case base, here from the csv file:
case_base = pycbr.casebase.SimpleCSVCaseBase(f.name)
# 2- Define the set of similarity functions. Here we we will use a linear similarity for the quantile of each value:
recovery = pycbr.recovery.Recovery([(x, pycbr.models.QuantileLinearAttribute()) for x in iris["feature_names"]])
# 3- Define the aggregation method to provide solutions. Here, we use the majoritary species:
aggregation = pycbr.aggregate.MajorityAggregate("species")
# 4- Create the CBR instance:
cbr = pycbr.CBR(case_base, recovery, aggregation, server_name="Iris-demo")

# The WSGI app should be exposed so it can be loaded with deployment servers
app = cbr.app

if __name__ == '__main__':
    # Start the development server if running as a script
    app.run()
