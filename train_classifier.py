import numpy as np

def train_nb(X, Y):
    n = X.shape[1]
    c_pos = X[Y == 1, :].sum(0)
    c_neg = X[Y == 0, :].sum(0)
    pi_pos = (1 + c_pos) / (n + c_pos.sum())
    pi_neg = (1 + c_neg) / (n + c_neg.sum())
    prior_pos = Y.mean()
    prior_neg = 1 - prior_pos
    w = np.log(pi_pos) - np.log(pi_neg)
    b = np.log(prior_pos) - np.log(prior_neg)
    return w, b

def inference_nb(X, w, b):
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels

data = np.loadtxt("movie_reviews/train.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
w, b = train_nb(X, Y)
np.savez("modelnb.npz", w, b)

predictions = inference_nb(X, w, b)
accuracy = (predictions == Y).mean()
print("accuracy: ", accuracy * 100)

f = open("movie_reviews/vocabulary.txt")
voc = f.read().split()
f.close()

ranking = np.argsort(w)
print("NEGATIVE WORDS")
for i in ranking[:20]:
    print(voc[i], w[i])
print()
print("POSITIVE WORDS")
for i in ranking[-20:]:
    print(voc[i], w[i])
