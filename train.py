import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns


def model_performance(y_true, y_pred):  # Returns Precision, Recall and Accuracy
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), accuracy_score(y_true, y_pred)


def generate_confusion_matrix(y_test, y_pred, model_name, vectorizer_name, filter_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Test Confusion Matrix')
    plt.savefig(f'model_results/confusion_matrices/{model_name}_{vectorizer_name}_{filter_name}.png',
                bbox_inches='tight')
    plt.close()


def generate_reports(vectorizers, models, filtered_datasets):
    # Preparing result dataframe
    cols = ['ML Model', 'Feature', 'Filter', 'Precision', 'Recall', 'Accuracy']
    result_df = pd.DataFrame(columns=cols)

    for filter_name, filtered_dataset in filtered_datasets.items():  # Filtered datasets include POS filtered text
        X, y = filtered_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
        for model_name, model in models.items():  # ML Models
            for vectorizer_name, vectorizer in vectorizers.items():  # Vectorizers are different ways of representing input text data
                # Vectorize data
                X_train_vector = vectorizer.fit_transform(X_train)
                X_test_vector = vectorizer.transform(X_test)

                model.fit(X_train_vector, y_train)  # Train Model

                y_pred = model.predict(X_test_vector)
                precision, recall, accuracy = model_performance(y_test, y_pred)
                # Record result in DataFrame
                result_df = result_df.append(pd.DataFrame([[model_name, vectorizer_name, filter_name,
                                                            precision, recall, accuracy]], columns=cols), ignore_index=True)

                generate_confusion_matrix(y_test, y_pred, model_name, vectorizer_name, filter_name)

    result_df.to_csv('model_results/result.csv')


vectorizers = {
    'TFIDF': TfidfVectorizer(min_df=10),
    'Frequency_Count': CountVectorizer(min_df=10)
}

models = {
    'Naive_Bayes': MultinomialNB(),
    'Logistic_Regression': LogisticRegression(max_iter=10000),
    'Random_Forest_30': RandomForestClassifier(n_estimators=30),
}

filtered_data = {
    'No_Filter': 'processed_data/processed_dataset.pkl',
    'Noun_Adjective': 'processed_data/pos_filtered/NN_JJ_dataset.pkl',
    'Noun': 'processed_data/pos_filtered/NN_dataset.pkl',
    'Noun_Verb': 'processed_data/pos_filtered/NN_VB_dataset.pkl',
    'Verb_Adjective': 'processed_data/pos_filtered/VB_JJ_dataset.pkl',
}


def data_2_xy(filters):  # Goes through filters dictionary and replaces data path with usable [X, y] for each
    y = pd.read_pickle(filters['No_Filter'])['label']
    for filter_name, path in filters.items():
        data = pd.read_pickle(path)
        if filter_name == 'No_Filter':
            X = data['processed_text'] + data['processed_title']  # To use both, concatenating Text and Title into one
        else:
            X = data['pos_processed_text'] + data['pos_processed_title']
        filters[filter_name] = [X, y]
    return filters


data_2_xy(filtered_data)
generate_reports(vectorizers, models, filtered_data)
