import os
import collections

from porter import stem

def remove_punctuation(s):
    for p in "!\"#%$&'()*+,-./:;<=>?@[\\]{|}~":
        s = s.replace(p, " ")
    return s

def read_stopwords(filename):
    stopwords = []
    f = open(filename, encoding="utf-8")
    text = f.read()
    f.close()
    for word in text.split():
        stopwords.append(word)
    return stopwords

def read_document(filename):
    f = open(filename, encoding="utf-8")
    text = f.read().lower()
    f.close()
    text = remove_punctuation(text)
    words = []
    stopwords = read_stopwords("movie_reviews/aclImdb/stopwords.txt")
    # print(stopwords)
    for w in text.split():
        if len(w) > 2 and w not in stopwords:
            words.append(stem(w))
    return words

def write_vocabulary(voc, filename, n):
    f = open(filename, "w")
    for word, count in voc.most_common(n):
        print(word, file=f)
    f.close()

voc = collections.Counter()
for f in os.listdir("movie_reviews/aclImdb/train/pos"):
    path = "movie_reviews/aclImdb/train/pos/" + f
    words = read_document(path)
    voc.update(words)
for f in os.listdir("movie_reviews/aclImdb/train/neg"):
    path = "movie_reviews/aclImdb/train/neg/" + f
    words = read_document(path)
    voc.update(words)
write_vocabulary(voc, "movie_reviews/vocabulary.txt", 2000)