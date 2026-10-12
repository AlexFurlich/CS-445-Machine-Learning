#!/usr/bin/env python3
"""
Simple Naive Bayes Text Classifier

Author: Mostly Claude Sonnet 4.0 with some edits by Nathan Sprague
"""

import csv
import re
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
import math


class NaiveBayesClassifier:
    """
    Simple Naive Bayes text classifier with Laplace smoothing.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Smoothing pseudo-count added to every word count.
                alpha=1 is standard Laplace (add-one) smoothing.  Must be
                greater than 0.
        """
        self.alpha = alpha
        self.class_priors = {}
        self.word_counts = {}
        self.total_words = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.classes = []

    def fit(self, X, y):
        """
        Train the classifier.

        Args:
            X (list): List of tokenized documents (list of words)
            y (list): List of class labels
        """
        # Store classes
        self.classes = sorted(set(y))
        n_samples = len(X)

        # Build vocabulary from all documents
        self.vocabulary = set()
        for doc in X:
            self.vocabulary.update(doc)

        # Calculate class priors
        class_counts = Counter(y)
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / n_samples

        # Class conditional probabilities
        self.word_probs = {class_label: {} for class_label in self.classes}

        for class_label in self.classes:
            # Get all documents for this class
            class_docs = [X[i] for i in range(len(X)) if y[i] == class_label]

            # Count word frequencies in this class
            word_counts = Counter()
            total_words = 0

            for doc in class_docs:
                word_counts.update(doc)
                total_words += len(doc)

            self.word_counts[class_label] = word_counts
            self.total_words[class_label] = total_words

            # Calculate smoothed probabilities: pretend every word in the
            # vocabulary was seen alpha extra times in this class
            denominator = total_words + self.alpha * len(self.vocabulary)
            for word in self.vocabulary:
                word_count = word_counts[word]
                self.word_probs[class_label][word] = (
                    word_count + self.alpha
                ) / denominator

    def predict_single(self, document):
        """
        Predict the class for a single document.

        Args:
            document (list): Tokenized document (list of words)

        Returns:
            str: Predicted class label
        """
        class_scores = {}

        # Words that never appeared in training have no probability
        # estimates, so they are skipped
        known_words = [word for word in document if word in self.vocabulary]

        # Calculate scores for each class
        for class_label in self.classes:
            # Start with log of class prior
            log_prob = math.log(self.class_priors[class_label])

            # Add log probabilities of known words
            for word in known_words:
                word_prob = self.word_probs[class_label][word]
                log_prob += math.log(word_prob)

            class_scores[class_label] = log_prob

        # Return class with highest score
        return max(class_scores, key=class_scores.get)

    def predict(self, X):
        """
        Predict classes for multiple documents.

        Args:
            X (list): List of tokenized documents

        Returns:
            list: List of predicted class labels
        """
        return [self.predict_single(doc) for doc in X]


def tokenize_text(text):
    """
    Simple tokenization function that:
    1. Converts to lowercase
    2. Removes punctuation and numbers
    3. Splits on whitespace
    4. Filters out very short words

    Args:
        text (str): Input text

    Returns:
        list: List of tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split on whitespace and filter out short words
    tokens = [word for word in text.split() if len(word) > 2]

    return tokens


def load_and_process_data(filepath):
    """
    Load and process a csv file containing labeled text. Labels in the first column, text in the second.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (tokenized_texts, labels)
    """
    labels = []
    texts = []

    # Read the comma-delimited file
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip()
                text = row[1].strip()
                labels.append(label)
                texts.append(text)

    # Tokenize all texts
    tokenized_texts = [tokenize_text(text) for text in texts]

    return tokenized_texts, labels


def main():
    """
    Main function to illustrate use of the classifier.
    """
    # Load and process data
    print("Loading and processing data...")
    X_train, y_train = load_and_process_data("data_rt_train.csv")
    X_test, y_test = load_and_process_data("data_rt_test.csv")

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set distribution: {Counter(y_train)}")
    print(f"Test set distribution: {Counter(y_test)}")

    # Train the Naive Bayes classifier
    print("\nTraining Naive Bayes classifier...")
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    print(f"Vocabulary size: {len(nb_classifier.vocabulary)}")
    print(f"Class priors: {nb_classifier.class_priors}")

    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = nb_classifier.predict(X_test)

    # Calculate and report results
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == "__main__":
    main()
