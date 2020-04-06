#!/usr/bin/env python3

import pycbr
from pycbr.models import *

# Define a case base from the csv file
case_base = pycbr.casebase.SimpleCSVCaseBase("stackoverflow.csv")
# Define the set of similarity functions
recovery = pycbr.recovery.Recovery([
    ("title", TextAttribute())
],
    algorithm="brute")

# Define the aggregation method
aggregation = pycbr.aggregate.MajorityAggregate("label")

# Create a CBR instance
cbr = pycbr.CBR(case_base, recovery, aggregation, server_name="Text-demo")

# Expose the WSGI app
app = cbr.app

if __name__ == '__main__':
    # Start the development server
    app.run()
