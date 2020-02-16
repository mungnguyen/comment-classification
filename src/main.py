import os
import string

from preprocess import cleanAndPreprocess
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

print("Begin Load file\n")
# Load data
dir = os.path.dirname(__file__)
dirName = os.path.join(dir, '../data/aclImdb/train')

dataset = load_files(dirName, "r", categories=["pos", "neg"], encoding="utf-8")

print("End Load file\n")

X, y = dataset.data, dataset.target

# Clean and pre-process
documents = []

print("Begin pre-processing\n")

# X1 = X[:10]
for x in X:
    documents.append(" ".join(cleanAndPreprocess(x)))

gitprint("End pre-processing\n")

print("Begin vectorize \n")
# Vectorize
vectorizer = CountVectorizer(min_df=5, max_df=0.7, encoding='utf-8', lowercase=False)
X = vectorizer.fit_transform(documents).toarray()

print("End vectorize \n")

print(f"------------------X------------------\n{documents[0]}\n-------\n{documents[1]}\n--------\n{X[2]}\n--------\n{X[3]}\n---------")

print(f"------------------Y values------------------\n{y[0]}\n-------\n{y[1]}\n--------\n{y[2]}\n--------\n{y[3]}\n---------")

print(f"--------Voca--------------\n{vectorizer.vocabulary_}\n-------------------------")

# Text clean and Pre-processing

