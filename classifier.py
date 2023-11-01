"""
Text classification based on Naive Bayes

This project uses the Naive Bayes algorithm to conduct text classification experiments.
This experiment is for text classification of news titles. According to the given news
tags, train a classification model using title text and labels, and then classify the
news title text of the test set.

Author: Klasnov Shao
Date: Nov. 1st, 2023
Environment: Python 3.11, scikit-learn 1.3.2
"""

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split('\t')
            dataset.append((text, label))
    return dataset

# Text data segmentation
def tokenize(text):
    return ' '.join(jieba.cut(text))

# Main function
def main():
    #Import experimental data
    train_data = load_dataset('data/train.txt')
    test_data = load_dataset('data/test.txt')

    # Participle
    train_data = [(tokenize(text), label) for text, label in train_data]
    test_data = [(tokenize(text), label) for text, label in test_data]

    # Text vectorization
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([text for text, _ in train_data])
    y_train = [label for _, label in train_data]
    X_test = vectorizer.transform([text for text, _ in test_data])
    y_test = [label for _, label in test_data]

    # Build and train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Perform model performance testing
    y_test_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    print()
    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
