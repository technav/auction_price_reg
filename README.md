Auction Price Regression
======================

Predict the sale price of a particular piece of heavy equipment at auction
based on it's usage, equipment type, and configuration.  The data is sourced
from auction result postings and includes information on usage and
equipment configurations.

Data
======================
The data are in `./data`. Train data is used to cross-validate and identify
potential models and test data is used to score those models.

Evaluation
======================
The evaluation of the model is based on Root Mean Squared Log Error.
