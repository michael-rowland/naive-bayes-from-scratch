from collections import defaultdict
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, k=0.5):
        self.k = k

    def prior_probabilities(self, y):
        count = defaultdict(int)
        for i in y:
            count[i] += 1
        return {i: [count[i], count[i]/len(y)] for i in count}

    def conditional_probabilities(self, X, y, k=0.5):
        # there has to be a better way to write this
        # establish blank dictionary
        counts = {i: [0] * len(X[0]) for i in list(set(y))}
        # cumulative count for each class
        for i in range(len(y)):
            for idx, count in enumerate(X[i]):
                counts[y[i]][idx] += count
        # turns cumulative count into "smoothed" probability
        for key, values in counts.items():
            total = sum(values)
            for idx, value in enumerate(values):
                values[idx] = (value + k) / (total + 2 * k)
        return counts

    def fit(self, X, y):
        self.priors = self.prior_probabilities(y)
        self.conds = self.conditional_probabilities(X, y, self.k)
        # class_map = {i: unique[i] for i in range(len(unique))}

    def predict(self):
        # PASS IN TOKENIZED VALUES, WORD PROBABILITIES, AND PRIORS
        # CALCULATE PROBABILITIES FOR EACH CLASSIFICATION
        # WRITE A INDIVIDUAL PREDICTION FUNCTION 
            # def score(dictionary of conditional probabiliity, prediction text_vector):
            # do I need log scoring?
            # spam_probability function from DS, from scratch book is good
        # SOMETHING LIKE: for each i (class) in dictionary of conditional probabilities
            # call score function, return max
        pass


def main():
    # text/spam processing done here
    pass


rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 10))
y = np.array([1, 2, 3, 1, 2, 3])
mnb = MultinomialNaiveBayes()
mnb.conditional_probabilities(X, y)
mnb.fit(X, y)
# print(mnb.prior_probabilities(y))

# TODO