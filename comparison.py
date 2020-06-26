import glob, re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes import MultinomialNaiveBayes

def preprocessing():
    path = 'data/*/*'
    X = []
    y = []

    for email in glob.glob(path):
        with open(email, errors='ignore') as file:
            for line in file:
                if line.startswith("Subject:"):
                    text = line.lstrip("Subject: ").strip('\n')
                    X.append(text)
                    y.append(1 if "spam" in email else 0)
                    break

    vect = CountVectorizer(stop_words='english')
    vect.fit(X)
    dtm = vect.transform(X)
    dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())
    X = dtm.values
    X_test_data = []
    with open('test.txt', errors='ignore') as file:
        for line in file:
            X_test_data.append(line.strip('\n'))
    X_test = vect.transform(X_test_data)
    X_test = pd.DataFrame(X_test.todense(), columns=vect.get_feature_names())
    X_test = X_test.values
    return X, y, X_test

def main():
    # TOY DATASET
    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 10))
    y = np.array([1, 2, 3, 1, 2, 3])
    X_test = np.array([5, 5, 4, 5, 3, 2, 1, 3, 1, 4])
    # personal implementation
    print("Personal Implementation (Toy Dataset)")
    print("-"*50)
    mnb = MultinomialNaiveBayes()
    mnb.conditional_probabilities(X, y)
    mnb.fit(X, y)
    mnb.predict(X_test)
    mnb.get_proba()

    # scikit-learn implementation
    print("Scikit-learn Implementation (Toy Dataset)")
    print("-"*50)
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X, y)
    X_test = X_test.reshape(1, -1)
    print(nb.predict(X_test))
    print(nb.predict_proba(X_test))

    # GATHER DATA FROM SPAM DATASET
    X, y, X_test = preprocessing()
    
    # personal implementation
    print("Personal Implementation (Spam Dataset)")
    print("-"*50)
    mnb = MultinomialNaiveBayes()
    mnb.conditional_probabilities(X, y)
    mnb.fit(X, y)
    mnb.predict(X_test)
    mnb.get_proba()

    # scikit-learn implementation
    print("Scikit-learn Implementation (Spam Dataset)")
    print("-"*50)
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X, y)
    print(nb.predict(X_test))
    print(nb.predict_proba(X_test))


if __name__ == "__main__": main()