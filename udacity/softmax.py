import numpy as np

def softmax(x):
    total_sum = np.exp(x).sum(axis=0)
    return np.exp(x) / total_sum

# Tests
scores = np.array([1, 2, 3])
probability_scores = softmax(scores)
print("Probability vector {}".format(probability_scores))
print("Probability vector norm {}".format(probability_scores.sum()))
scores = np.array([[1, 2, 3, 6], [2, 4, 5, 6], [3, 8, 7, 6]])
probability_scores = softmax(scores)
print("Probability vector {}".format(probability_scores))
print("Probability vector norm {}".format(probability_scores.sum(axis=0)))
sum = 1000000000
for x in range(1000000):
    sum += 0.000001