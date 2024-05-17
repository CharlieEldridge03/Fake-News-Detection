import pickle
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import shap
import torch
import transformers
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.utils import resample
import joblib
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from scipy.special import softmax


def load_liar_data():
    LIAR_train_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_train.csv",
                                       index_col=False).dropna()
    LIAR_test_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_test.csv",
                                      index_col=False).dropna()
    LIAR_valid_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_valid.csv",
                                       index_col=False).dropna()

    X_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[0: 22]]
    y_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[22]]

    X_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[0: 22]]
    y_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[22]]

    X_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[0: 22]]
    y_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[22]]

    # Combining the test/validation sets to create a 25% train/test split
    X_test = pd.concat([X_test, X_valid], ignore_index=True)
    y_test = pd.concat([y_test, y_valid], ignore_index=True)

    return X_train, X_test, y_train, y_test


def load_isot_text_data():
    ISOT_text_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_texts.csv",
        index_col=False).dropna()

    X = ISOT_text_dataframe[ISOT_text_dataframe.columns[0: 22]]
    y = ISOT_text_dataframe[ISOT_text_dataframe.columns[22]]

    # 25% Train/Test Split to reflect LIAR dataset split
    # I had to use 20% due to stratified splitting resulting in 30% actual split size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test


def load_isot_title_data():
    ISOT_title_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_titles.csv",
        index_col=False).dropna()

    X = ISOT_title_dataframe[ISOT_title_dataframe.columns[0: 22]]
    y = ISOT_title_dataframe[ISOT_title_dataframe.columns[22]]

    # 25% Train/Test Split to reflect LIAR dataset split
    # I had to use 20% due to stratified splitting resulting in 30% actual split size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test


def min_max_scale(X_train, X_test):
    min_max_scaler = MaxAbsScaler()

    scaled_X_train = min_max_scaler.fit_transform(X_train)
    scaled_X_test = min_max_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def create_tfidf_features(X_train, X_test, ngram_range):
    tfidf = TfidfVectorizer(ngram_range=ngram_range)
    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)

    scaled_X_train, scaled_X_test = min_max_scale(X_train_tfidf, X_test_tfidf)

    return scaled_X_train, scaled_X_test


def oversample(X_train, y_train):
    over_sampler = SMOTE(random_state=42)
    X_train, y_train = over_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def perform_truncation(X_train, X_test, y_train, num_components):
    truncate = TruncatedSVD(n_components=num_components, n_iter=5, random_state=42)
    X_train_truncated = truncate.fit_transform(X_train, y_train)
    X_test_truncated = truncate.transform(X_test)

    X_train_truncated_df = pd.DataFrame(X_train_truncated, columns=truncate.get_feature_names_out())
    X_test_truncated_df = pd.DataFrame(X_test_truncated, columns=truncate.get_feature_names_out())

    min_max_scaler = MinMaxScaler()
    scaled_X_train = min_max_scaler.fit_transform(X_train_truncated_df)
    scaled_X_test = min_max_scaler.transform(X_test_truncated_df)

    return scaled_X_train, scaled_X_test


def main():
    """ SVC Voting Classifier Ensemble Using TFIDF Features """
    X_train, X_test, y_train, y_test = load_liar_data()
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test

    X_train, X_test = create_tfidf_features(X_train, X_test, (1, 1))
    print("X_train Shape: {}".format(X_train.shape))

    X_train, y_train = oversample(X_train, y_train)

    rbf_svc = SVC(C=1, gamma=0.1, decision_function_shape='ovo', kernel="rbf", random_state=42)
    linear_svc = SVC(C=1, gamma=0.1, decision_function_shape='ovo', kernel="linear", random_state=42)
    sigmoid_svc = SVC(C=1, gamma=0.1, decision_function_shape='ovo', kernel="sigmoid", random_state=42)
    ensemble_svc = VotingClassifier(estimators=[('rbf_svc', rbf_svc), ('linear_svc', linear_svc), ('sigmoid_svc', sigmoid_svc)], voting='hard', verbose=1)

    ensemble_svc.fit(X_train, y_train)
    y_pred = ensemble_svc.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_svc.classes_)
    disp.plot()
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    plt.title("Accuracy Score: {}%".format(accuracy))
    plt.savefig("VOTING_ENSEMBLE_LIAR_TFIDF_oversampled.png")
    plt.show()

    """ SVC Stacking Classifier Ensemble Using TFIDF Features """
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train
    # X_test = X_test
    # y_train = y_train
    # y_test = y_test
    #
    # X_train, X_test = create_tfidf_features(X_train, X_test, (1, 1))
    # print("X_train Shape: {}".format(X_train.shape))
    #
    # X_train, y_train = oversample(X_train, y_train)
    #
    # rbf_svc = SVC(C=10, gamma=0.1, decision_function_shape='ovo', kernel="rbf", random_state=42, class_weight='balanced')
    # linear_svc = SVC(C=10, gamma=0.1, decision_function_shape='ovo', kernel="linear", random_state=42, class_weight='balanced')
    # sigmoid_svc = SVC(C=10, gamma=0.1, decision_function_shape='ovo', kernel="sigmoid", random_state=42, class_weight='balanced')
    # ensemble_svc = StackingClassifier(estimators=[('sigmoid_svc', sigmoid_svc), ('rbf_svc', rbf_svc)], final_estimator=linear_svc, verbose=1)
    #
    # ensemble_svc.fit(X_train, y_train)
    # y_pred = ensemble_svc.predict(X_test)
    #
    # print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_svc.classes_)
    # disp.plot()
    # accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    # plt.title("Accuracy Score: {}%".format(accuracy))
    # plt.savefig("STACKING_ENSEMBLE_LIAR_TFIDF_oversampled.png")
    # plt.show()


if __name__ == "__main__":
    main()
