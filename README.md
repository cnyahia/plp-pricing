## Peak-Load Pricing and Demand Management for Ridesourcing Platforms
This repository contains material for analyzing a pricing mechanism that maximizes platform revenue while staggering predicted peaks in the demand-supply mismatch.
The pricing mechanism is applied to ridesourcing data describing Lyft operations in NYC. A csv file with the cleaned data is available in this repository.

### Overview
  * network.py: main script for time-dependent implementation of the proposed mechanism and for maintaining the defined stochastic and deterministic processes across time
  * odpair.py: a class that represents functions needed per origin-destination pair. For previously observed and predicted (future) rides between the o-d pair, we evaluate the number of starts or ends that are anticipated within the upcoming time horizon
  * region.py: aggregates info. across o-d pairs, and implements the proposed convex optimization program using cvxpy
  * utils.py: contains utility functions for integrating and evaluating empirical distributions, computing the arrival rate maximum likelihood estimator, and processing the data.
  
