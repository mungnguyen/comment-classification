import os
import string

from pathlib import Path
from datetime import datetime
from preprocess import cleanAndPreprocess

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report as metric

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB

print("Begin Load file\n")

dirPath = Path(os.path.dirname(__file__))
print(dirPath.name)

trainDir = os.path.join(dirPath.parent, './data/aclImdb/train')
testDir = os.path.join(dirPath.parent, './data/aclImdb/test')

X_train = []
y_train = []
X_test = []
y_test = []
train_docs = []
test_docs = []

# Load data
if os.path.isfile('train-docs-after-processing.txt'):
    train_file = open("./train-docs-after-preprocessing.txt", "r")
    train_docs = train_file.readlines()
    print(f"Train-docs length: {len(train_docs)}")

    train_label = open("./train-label-after-preprocessing.txt", "r")
    y_train = train_label.readlines()
    print(f"Train-label length: {len(y_train)}")
else:
    #Load train file
    trainset = load_files(trainDir, "r", categories=["pos", "neg"], encoding="utf-8")
    X_train, y_train = trainset.data, trainset.target

    # Clean and pre-process
    print("Begin pre-processing trainset\n")

    for x in X_train:
        train_docs.append(" ".join(cleanAndPreprocess(x)))
    
    print("End pre-processing trainset\n")

    # Print to file
    with open('train-docs-after-preprocessing.txt', 'w') as f_train:
        for item in train_docs:
            f_train.write("%s\n" % item)
        print("print train_docs to file successly")

    with open('train-label-after-preprocessing.txt', 'w') as f_train_label:
        for item in y_train:
            f_train_label.write("%s\n" % item)
        print("print train_lable to file successly")

if os.path.isfile('test-docs-after-processing.txt'): 
    test_file = open("./test-docs-after-preprocessing.txt", "r")
    test_docs = test_file.readlines()
    print(f"Test-docs length: {len(test_docs)}")
else:
    #Load test file
    print("Begin load test file")
    testset = load_files(testDir, "r", categories=["pos", "neg"], encoding="utf-8")
    X_test, y_test = testset.data, testset.target
    print("End Load test file\n")

    # Clean and pre-process
    print("Begin pre-processing testset\n")
    for x in X_test:
        test_docs.append(" ".join(cleanAndPreprocess(x)))
    print("End pre-processing testset\n")

    # Print to file
    with open('test-docs-after-preprocessing.txt', 'w') as f_test:
        for item in test_docs:
            f_test.write("%s\n" % item)
        print("print test_docs to file successly") 

    with open('test-label-after-preprocessing.txt', 'w') as f_test_label:
        for item in y_test:
            f_test_label.write("%s\n" % item)
        print("print test_label to file successly")   

# Vectorize
print("Begin vectorize \n")
# IF-IDF
print("Begin tf-idf")
train_tfidf = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, encoding='utf-8', lowercase=False)
X_train = train_tfidf.fit_transform(train_docs)

print(f"Voca len: {len(train_tfidf.vocabulary_)}")

print("Begin test tf-idf")
test_tf_idf = TfidfVectorizer(min_df=5, max_df=0.7, encoding='utf-8', lowercase=False, vocabulary=train_tfidf.vocabulary_)
X_test = test_tf_idf.fit_transform(test_docs)
print("End tf-idf")

print("End vectorize \n")

# Training data - testing data
    # Traning with Decision Tree
print("Begin training data with decision tree")
start_time = datetime.now()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

end_time = datetime.now()
print(f"Time to training: {end_time - start_time}")

print("End training data with decision tree")

    # Testing with decision tree
print("Begin testing data with decision tree")
start_time = datetime.now()

dstree_pred = clf.predict(X_test)

result = metric(y_test, dstree_pred)

print(f"{result}\n")

end_time = datetime.now()
print(f"Time to testing: {end_time - start_time}")

print("End testing data with decision tree")

    # Traning with Random Forest
print("Begin training with Random Forest")
start_time = datetime.now()

clf = RandomForestClassifier(n_estimators=1000, random_state=0)
clf.fit(X_train, y_train) 

end_time = datetime.now()
print(f"Time to training: {end_time - start_time}")

print("End training with Random Forest")

    # Tesing with Random Forest
print("Begin testing with Random Forest")
start_time = datetime.now()

rd_pred = clf.predict(X_test)

result = metric(y_test, rd_pred)

print(f"{result}\n")

end_time = datetime.now()
print(f"Time to testing: {end_time - start_time}")

print("End testing with Random Forest")

    # Traning with SVM
print("Begin training with SVM")
start_time = datetime.now()

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

end_time = datetime.now()
print(f"Time to training: {end_time - start_time}")

print("End training with SVM")

    # Tesing with SVM
print("Begin testing with SVM")
start_time = datetime.now()

svm_pred = clf.predict(X_test)

result = metric(y_test, svm_pred)

end_time = datetime.now()
print(f"Time to testing: {end_time - start_time}")

print(f"{result}\n")

print("End testing with SVM")

    # Traning with Rocchio Classification
print("Begin training with Rocchio Classification")
start_time = datetime.now()

clf = NearestCentroid()
clf.fit(X_train, y_train)

end_time = datetime.now()
print(f"Time to training: {end_time - start_time}")

print("End training with Rocchio Classification")

    # Tesing with Rocchio Classification
print("Begin testing with Rocchio Classification")
start_time = datetime.now()

rc_pred = clf.predict(X_test)

result = metric(y_test, rc_pred)

print(f"{result}\n")

end_time = datetime.now()
print(f"Time to testing: {end_time - start_time}")

print("End testing with Rocchio Classification")

    # Traning with Naive Bayes Classifier
print("Begin training with Naive Bayes Classifier, use Gaussian Naive Bayes algorithm")
start_time = datetime.now()

clf = GaussianNB()
clf.fit(X_train, y_train)

end_time = datetime.now()
print(f"Time to training: {end_time - start_time}")

print("End training with Naive Bayes Classifier")

    # Tesing with Naive Bayes Classifier
print("Begin testing with Naive Bayes Classifier")
start_time = datetime.now()

nb_pred = clf.predict(X_test)

precision, recall, f1 = precision_recall_fscore_support(y_test, nb_pred)

result = metric(y_test, nb_pred)

end_time = datetime.now()
print(f"Time to testing: {end_time - start_time}")

print(f"{result}\n")

print("End testing with Naive Bayes Classifier")

print("End programe")
