from operator import xor
import numpy as np
import os
from porter import stem


def load_vocabulary(filename):
    f = open(filename)
    words = f.read().split()
    f.close()
    voc = {}
    index = 0
    for word in words:
        voc[word] = index
        index += 1
    return voc

def remove_punctuation(s):
    for p in "!\"#%$&'()*+,-./:;<=>?@[\\]{|}~":
        s = s.replace(p, " ")
    return s

def extract_features(filename, voc):
    f = open(filename, encoding="utf-8")
    text = f.read().lower()
    f.close()
    text = remove_punctuation(text)
    bow = np.zeros(len(voc))
    for word in text.split():
        word = stem(word)
        if word in voc:
            index = voc[word]
            bow[index] += 1
    return bow

def inference_nb(X, w, b):
    logits = X @ w + b
    # print(logits)
    labels = (logits > 0).astype(int)
    return labels, logits

voc = load_vocabulary("movie_reviews/vocabulary.txt")
features = []
labels = []
files = []
for f in os.listdir("movie_reviews/aclImdb/validation/pos"):
    path = "movie_reviews/aclImdb/validation/pos/" + f
    bow = extract_features(path, voc)
    features.append(bow)
    labels.append(1)
    files.append(path)
for f in os.listdir("movie_reviews/aclImdb/validation/neg"):
    path = "movie_reviews/aclImdb/validation/neg/" + f
    bow = extract_features(path, voc)
    features.append(bow)
    labels.append(0)
    files.append(path)
    


data = np.load("modelnb.npz")
w = data["arr_0"]
b = data["arr_1"]
X = np.stack(features)
Y = np.array(labels)

# print("w: ", w)
# print("b: ", b)
# print("shape: ", X.shape)

predictions, logits = inference_nb(X, w, b)
accuracy = (Y == predictions).mean()
print("test acc: ", accuracy * 100)

errs = xor(Y, predictions)
logit_errs = np.multiply(errs, logits)

corrs = np.logical_not(errs)
logit_corrs = np.multiply(corrs, logits)

ranking_errs = np.argsort(abs(logit_errs))
reviews_errs = []
for i in ranking_errs[-3:]:
    print(i, files[i], logit_errs[i])
    f = open(files[i], encoding="utf-8")
    rev = f.read()
    # print(rev)
    f.close()
print("\n\n\n\n")
ranking_corrs = np.argsort(abs(logit_corrs))
reviews_corrs = []
for i in ranking_corrs[-3:]:
    print(i, files[i], logit_corrs[i])
    f = open(files[i], encoding="utf-8")
    rev = f.read()
    # print(rev)
    f.close()