# Home Assignment 4

Name: Maria Savostianova
Student ID: 700793417
Course: CS5760 Natural Language Processing
Semester: Fall 2026

## Files

- hw4.py contains both programming tasks.
- Maria_Savostianova_HW4.docx contains my written answers and calculations.

## How to Run

The program uses Python, NLTK, and NumPy.

Install the libraries:

```bash
python3 -m pip install nltk numpy
```

Run the program:

```bash
python3 hw4.py
```

Internet access is needed to download the English stopword list the first time.

## Q1: NLP Preprocessing

This function processes the sentence:
"NLP techniques are used in virtual assistants like Alexa and Siri."

It splits the sentence into tokens, removes English stopwords, and applies stemming. It prints the results after each step.

I used TreebankWordTokenizer for tokenization, the NLTK English stopword list, and PorterStemmer for stemming. Punctuation stays because the instructions only ask for stopword removal.

### Results

```text
Original Tokens:
['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']

Tokens Without Stopwords:
['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri', '.']

Stemmed Words:
['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri', '.']
```

Stemming can produce incomplete words, such as "techniqu." Lemmatization uses vocabulary and grammar to find a dictionary form. For "running," stemming gives "run," and lemmatization also gives "run" when treated as a verb.

Removing stopwords can reduce noise in document classification. However, removing "not" changes "not great" to "great," which can harm sentiment analysis.

## Q2: Scaled Dot-Product Attention

The function follows these steps:

1. Multiply Q by the transpose of K.
2. Divide the scores by the square root of the key dimension.
3. Apply softmax to each row to get attention weights.
4. Multiply the attention weights by V.

Before applying softmax, I subtract the maximum score in each row to keep the calculation numerically stable. This does not change the result.

The program uses the Q, K, and V matrices provided in the assignment.

### Results

```text
Attention Weights:
[[0.73105858 0.26894142]
 [0.26894142 0.73105858]]

Final Output:
[[2.07576569 3.07576569 4.07576569 5.07576569]
 [3.92423431 4.92423431 5.92423431 6.92423431]]
```

Dividing by the square root of the key dimension keeps the scores from becoming too large. This prevents softmax from becoming too extreme and helps keep gradients useful during training.

Self-attention compares each token's query with other tokens' keys to determine relevance. It then combines their values using attention weights to create an updated representation.

## Written Work

The Word document includes Part I short answers, Part II short answers, and the Part III calculation.

For Part III, I followed the lecture's unscaled dot-product method and rounded the attention weights to one decimal place before multiplying the Value vectors.

Rounded weights: [0.0, 0.9, 0.1]

Third token's new representation: [2.0, 7.8, 0.3]