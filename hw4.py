import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the English stopword list.
nltk.download("stopwords")


def preprocess(sentence):
    # Split the sentence into words and punctuation.
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(sentence)
    print("Original Tokens:", tokens)

    # Remove stopwords, ignoring capitalization.
    stop_words = set(stopwords.words("english"))
    filtered = [
        token for token in tokens
        if token.lower() not in stop_words
    ]
    print("Tokens Without Stopwords:", filtered)

    # Reduce the remaining words to stems.
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in filtered]
    print("Stemmed Words:", stemmed)


sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
preprocess(sentence)

import numpy as np


def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute Q multiplied by the transpose of K.
    scores = Q @ K.T

    # Step 2: Divide by the square root of the key dimension.
    d = K.shape[1]
    scaled_scores = scores / np.sqrt(d)

    # Step 3: Apply softmax separately to each row.
    # Subtracting the row maximum keeps the calculation stable.
    exp_scores = np.exp(
        scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
    )
    weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Step 4: Multiply the attention weights by V.
    output = weights @ V

    return weights, output


# Test inputs provided in the assignment.
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

weights, output = scaled_dot_product_attention(Q, K, V)

print("\nAttention Weights:")
print(weights)

print("\nFinal Output:")
print(output)