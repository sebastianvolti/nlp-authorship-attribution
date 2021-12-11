import matplotlib.pyplot as plt
import nltk

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')


def tokenize_text(text_path):
    with open(text_path, 'r') as text:
        return word_tokenize(text.read())


def get_frecuencies(tokenized_text):
    return FreqDist(word.lower() for word in tokenized_text)


def plot(data):
    plots = []
    plt.xscale('log')
    plt.yscale('log')
    for tokens_ranges, frecuencies, label in data:
        plots.append(plt.plot(tokens_ranges, frecuencies, label=label)[0])

    plt.legend(handles=plots)
    plt.show()
