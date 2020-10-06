## Peak-Load Pricing and Demand Management for Ridesourcing Platforms
This repository contains material for analyzing a pricing mechanism that maximizes platform revenue while staggering peaks in the load process. In this case, the load process describes the predicted demand-supply mismatch, and the *objective* is to influence passengers to depart during off-peak time periods. We use a multinomial logit choice model to represent user decisions.\
The pricing mechanism is applied to ridesourcing data from Lyft operations in NYC. A csv file with the cleaned data is available in this repository.

### Overview
  * network.py: main script for time-dependent implementation of the proposed mechanism and for maintaining the defined stochastic and deterministic processes across time
  * odpair.py: a class that represents functions needed per origin-destination pair. For previously observed and predicted (future) rides between the o-d pair, we evaluate the number of starts or ends that are anticipated within the upcoming time horizon
  * region.py: a class that aggregates info. across o-d pairs and implements the proposed convex optimization program using cvxpy
  * utils.py: contains utility functions for integrating/evaluating empirical distributions, computing the arrival rate maximum likelihood estimator, and processing the data.
  
