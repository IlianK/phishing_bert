import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Load data
def load_preprocessed_data(train_load_path, test_load_path):
    df_train = pd.read_csv(train_load_path)
    df_test = pd.read_csv(test_load_path)
    return df_train, df_test


# Load trained model
def load_model(models_folder, model_name):
    model_path = os.path.join(models_folder, f"{model_name.replace(' ', '_').lower()}.pkl")
    return joblib.load(model_path)


# Apply scaling conditionally
def conditional_scaling(X_train, X_val, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    return X_train, X_val


# Train and save models
def train_and_save_model(model, param_grid, X_train, y_train, model_name, models_folder, scale_features=False):
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(models_folder, f"{model_name}.pkl"))

    print(f"Best {model_name} params:", grid_search.best_params_)
    return best_model


# Evaluate model
def evaluate_model(model_name, test_file, models_folder):
    df_test = pd.read_csv(test_file)
    y_test = df_test['label']
    X_test = df_test.drop(columns=['label'])

    model = load_model(models_folder, model_name)
    vectorizer_path = os.path.join(models_folder, "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)

    X_test_text = X_test['body']
    X_test_numerical = X_test.drop(columns=['body'])

    X_test_tfidf = vectorizer.transform(X_test_text)

    if model_name in ["log_regression"]:
        _, X_test_numerical_scaled = conditional_scaling(X_test_numerical, X_test_numerical, scale=True)
    else:
        _, X_test_numerical_scaled = conditional_scaling(X_test_numerical, X_test_numerical, scale=False)

    X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_numerical_scaled])
    y_pred = model.predict(X_test_combined)

    print(f"\nEvaluation for {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    display(report_df)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Phishing'], yticklabels=['Ham', 'Phishing'])
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
