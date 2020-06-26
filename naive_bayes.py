from collections import defaultdict
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, k=0.5):
        self.k = k

    def prior_probabilities(self, y):
        count = defaultdict(int)
        for i in y:
            count[i] += 1
        return {i: count[i]/len(y) for i in count}

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
        # calculates prior/conditional probabilities for each classification
        self.priors = self.prior_probabilities(y)
        self.conds = self.conditional_probabilities(X, y, self.k)

    def class_probability(self, word_counts, cond_probs, prior):
        # multiplies the two lists by each other, sums them, multiplies by prior
        products = [a*b for a, b in zip(word_counts, cond_probs)]
        return np.log(sum(products) * prior)

    def predict(self, word_counts):
        self.results = {}
        for i, probs in self.conds.items():
            self.results[i] = self.class_probability(
                word_counts,
                probs,
                self.priors[i]
            )
        # print(max(self.results, key=self.results.get))
        # return max(self.results, key=self.results.get)

    def get_proba(self):
        print(self.results)
        return self.results
