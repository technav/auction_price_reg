Regression Case Study
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

Note that this loss function is sensitive to the *ratio* of predicted values to
the actual values, a prediction of 200 for an actual value of 100 contributes
approximately the same amount to the loss as a prediction of 2000 for an actual
value of 1000.  To convince yourself of this, recall that a difference of
logarithms is equal to a single logarithm of a ratio, and rewrite each summand
as a single logarithm of a ratio.
