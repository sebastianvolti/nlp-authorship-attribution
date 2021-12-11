from typing import OrderedDict
import nltk
import numpy as np
import os
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

nltk.download('averaged_perceptron_tagger')


### PARAMETERS ###

TRAINING_PROPORTION = 0.8
TOP_WORDS_BOW = 150


### PATHS ###

DATASET_PATH = "dataset/"


### AUX FUNCTIONS ###

def print_results(score):
    print("Accuracy: {0:.2f}%".format(score * 100))


def train_and_predict(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)


def normalize_features(features):
    return features / np.c_[np.apply_along_axis(np.linalg.norm, 1, features)]


def get_bow_features(tokens, texts, top_words_amount=TOP_WORDS_BOW, normalize=False):
    fdist = nltk.FreqDist(tokens)
    vocab = [v[0] for v in sorted(fdist.items(), key=lambda x: x[1], reverse=True)][:top_words_amount]
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
    
    fvs_bow = vectorizer.fit_transform(texts).toarray().astype(np.float64)

    if normalize:
        fvs_bow = normalize_features(fvs_bow)

    return fvs_bow


def get_pos_features(texts, normalize=False):
    texts_pos = [[p[1] for p in nltk.pos_tag(nltk.word_tokenize(text))] for text in texts]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    
    fvs_syntax = np.array([[text.count(pos) for pos in pos_list] for text in texts_pos]).astype(np.float64)

    if normalize:
        fvs_syntax = normalize_features(fvs_syntax)

    return fvs_syntax


def read_dataset_files(dataset_path):
    all_texts_with_classes = []
    filenames = []

    for filename in os.listdir(dataset_path):
        filenames.append(filename)
        file_path = dataset_path + filename
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                author = re.search("\D+(?=\d+\.txt)", filename).group(0)
                all_texts_with_classes.append((text, author))

    return all_texts_with_classes, filenames


def extract(texts_with_classes):
    all_texts = [t[0] for t in texts_with_classes]
    classes = [t[1] for t in texts_with_classes]
    all_text = "".join([t[0] for t in texts_with_classes])
    all_tokens = nltk.word_tokenize(all_text)

    return all_texts, classes, all_tokens


### MAIN CODE ###

# Training - Process text files
print()
print("Reading source files...\n")
all_texts_with_classes, _ = read_dataset_files(DATASET_PATH)

print("--- PROCESSING TRAINING DATA (using {0:.0f}% of the dataset)\n".format(TRAINING_PROPORTION * 100))
print("Extracting data from training texts...\n")
training_texts = all_texts_with_classes[:int(TRAINING_PROPORTION * len(all_texts_with_classes))]
all_texts, classes, all_tokens = extract(training_texts)

# Training - Get features
fvs = OrderedDict()
print("Getting training bow features...\n")
bow = get_bow_features(all_tokens, all_texts)
print("Getting training pos features...\n")
pos = get_pos_features(all_texts)

print("Getting training normalized features...\n\n")
fvs['both'] = normalize_features(np.c_[bow, pos])
fvs['bow'] = normalize_features(bow)
fvs['pos'] = normalize_features(pos)


# Test - Process text files
print("--- PROCESSING TEST DATA (using {0:.0f}% of the dataset)\n".format((1 - TRAINING_PROPORTION) * 100))
print("Extracting data from test texts...\n")
test_texts = all_texts_with_classes[:int((1 - TRAINING_PROPORTION) * len(all_texts_with_classes))]
all_texts_test, test_classes_test, all_tokens_test = extract(test_texts)

# Test - Get features
fvs_test = OrderedDict()
print("Getting test bow features...\n")
bow = get_bow_features(all_tokens_test, all_texts_test)
print("Getting test pos features...\n")
pos = get_pos_features(all_texts_test)

print("Getting test normalized features...\n\n")
fvs_test['bow'] = normalize_features(bow)
fvs_test['pos'] = normalize_features(pos)
fvs_test['both'] = normalize_features(np.c_[bow, pos])


# Predict authors
results = OrderedDict()
print(f"-------------- RESULTS -----------------\n")
for feature_type in fvs.keys():
    results[feature_type] = [
        "{0:.2f}%".format(train_and_predict(LinearSVC(), fvs[feature_type], classes, fvs_test[feature_type], test_classes_test) * 100),
        "{0:.2f}%".format(train_and_predict(GaussianNB(), fvs[feature_type], classes, fvs_test[feature_type], test_classes_test) * 100),
        "{0:.2f}%".format(train_and_predict(KNeighborsClassifier(), fvs[feature_type], classes, fvs_test[feature_type], test_classes_test) * 100)
    ]

dfObj = pd.DataFrame.from_dict(results, orient='index', columns=['SVM', 'NB', 'KNN']) 
print(dfObj)


print(f"----------------------------------------\n")
