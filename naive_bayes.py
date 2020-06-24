from sklearn.datasets import make_blobs

''' SCIKIT LEARN METHODS
fit(self, X, y[, sample_weight])
Fit Naive Bayes classifier according to X, y

get_params(self[, deep])
Get parameters for this estimator.

predict(self, X)
Perform classification on an array of test vectors X.

predict_log_proba(self, X)
Return log-probability estimates for the test vector X.

predict_proba(self, X)
Return probability estimates for the test vector X.

score(self, X, y[, sample_weight])
Return the mean accuracy on the given test data and labels.
'''

class MultinomialNaiveBayes:
    def __init__(self):
        # any/all setup inputs go here
        pass

    def fit(self, X_train):
        # TODO
        # 1. CALCULATE PROBABILITIES FOR EACH CLASS (PRIORS)
        # 2. CALCULATE CONDITIONAL PROBABILITIES FOR EACH CONDITION (CLASSIFICATION)
        # 3. ADD IN PSEUDOCOUNTS / SMOOTHING
        # returns nothing
        pass

    def predict(self):
        # PASS IN TOKENIZED VALUES, WORD PROBABILITIES, AND PRIORS
        # CALCULATE PROBABILITIES FOR EACH CONDITION
        pass

def fit_distribution():

    return

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
print(X.shape, y.shape)
print(X[:5])
print(y[:5])

