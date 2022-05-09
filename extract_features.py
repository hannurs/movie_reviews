from matplotlib.pyplot import stem
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

voc = load_vocabulary("movie_reviews/vocabulary.txt")
features = []
labels = []
for f in os.listdir("movie_reviews/aclImdb/train/pos"):
    path = "movie_reviews/aclImdb/train/pos/" + f
    bow = extract_features(path, voc)
    features.append(bow)
    labels.append(1)
for f in os.listdir("movie_reviews/aclImdb/train/neg"):
    path = "movie_reviews/aclImdb/train/neg/" + f
    bow = extract_features(path, voc)
    features.append(bow)
    labels.append(0)

X = np.stack(features)
Y = np.array(labels)
data = np.concatenate([X, Y[:, None]], 1)
# print(data.shape)
np.savetxt("movie_reviews/train.txt.gz", data)
# print(features[0])
# print(labels[0])

# print(bow)