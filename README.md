# simple-sentiment-analysis

## A quick guide to get your started with sentiment analysis using Machine Learning using python 3

This simple script, along with the training data provided (train.tsv) will allow you to quickly train a sentiment
analysis model.

The training data comes from movie reviews which excerpts of the review and the star rating given.

The model uses grid search to iterate through all the combinations of the parameters supplied and prints out the best
combination to use, with a little modification these can be stored as a json object and used to train future models.

Requires Pandas and Scikit-Learn