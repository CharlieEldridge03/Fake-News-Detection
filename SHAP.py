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
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import joblib
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from scipy.special import softmax
import warnings


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
    feature_names = tfidf.get_feature_names_out()

    return scaled_X_train, scaled_X_test, feature_names


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


def score_and_visualize(text, pipeline):
    prediction = pipeline([text])
    print(prediction[0])

    explainer = shap.Explainer(pipeline)
    shap_values = explainer([text])

    print(shap_values)

    shap.summary_plot(shap_values[0])
    shap.summary_plot(shap_values[1])
    shap.summary_plot(shap_values[2])
    shap.summary_plot(shap_values[3])
    shap.summary_plot(shap_values[4])
    shap.summary_plot(shap_values[5])


def main():
    """ SVC Stacked Model Ensemble Using Linguistic Features """
    X_train, X_test, y_train, y_test = load_liar_data()
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test

    X_train = X_train.drop(["text"], axis=1)
    X_test = X_test.drop(["text"], axis=1)
    print("X_train Shape: {}".format(X_train.shape))

    rbf_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="rbf", probability=True, random_state=42)
    linear_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="linear", probability=True, random_state=42)
    poly_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="poly", probability=True, random_state=42)
    sigmoid_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="sigmoid", probability=True, random_state=42)

    ensemble_svc_sc = StackingClassifier(estimators=[('linear_svc', linear_svc), ('poly_svc', poly_svc), ('sigmoid_svc', sigmoid_svc)], final_estimator=rbf_svc, verbose=1)
    ensemble_svc_sc.fit(X_train, y_train)
    y_pred = ensemble_svc_sc.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_svc_sc.classes_)
    disp.plot()
    plt.show()


    """ SVC VotingClassifier Ensemble Using Linguistic Features """
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train
    # X_test = X_test
    # y_train = y_train
    # y_test = y_test
    #
    # X_train = X_train.drop(["text"], axis=1)
    # X_test = X_test.drop(["text"], axis=1)
    # print("X_train Shape: {}".format(X_train.shape))
    #
    # # X_train, X_test = perform_truncation(X_train, X_test, y_train, 100)
    # # print("X_train Shape after TruncatedSVD: {}".format(X_train.shape))
    #
    # rbf_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="rbf", probability=True, random_state=42)
    # linear_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="linear", probability=True, random_state=42)
    # poly_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="poly", probability=True, random_state=42)
    # sigmoid_svc = SVC(C=1, gamma=1, decision_function_shape='ovo', kernel="sigmoid", probability=True, random_state=42)
    # ensemble_svc_vc = VotingClassifier(estimators=[('rbf_svc', rbf_svc), ('linear_svc', linear_svc), ('poly_svc', poly_svc), ('sigmoid_svc', sigmoid_svc)], voting='soft', verbose=1)
    #
    # ensemble_svc_vc.fit(X_train, y_train)
    # y_pred = ensemble_svc_vc.predict(X_test)
    #
    # print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_svc_vc.classes_)
    # disp.plot()
    # plt.show()

    """ Decision Plots Using TFIDF Features """
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train[:40]
    # X_test = X_test[:40]
    # y_train = y_train[:40]
    # y_test = y_test[:40]
    #
    # X_train, X_test, feature_names = create_tfidf_features(X_train, X_test, (1, 1))
    # X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
    # X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
    # print("X_train Shape: {}".format(X_train.shape))
    #
    # # X_train, X_test = perform_truncation(X_train, X_test, y_train, 100)
    # print("X_train Shape after TruncatedSVD: {}".format(X_train.shape))
    #
    # model = SVC(C=1, decision_function_shape='ovo', gamma=0.1, probability=True, random_state=42)
    # model.fit(X_train, y_train)
    #
    # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # shap_values = explainer.shap_values(X_test)
    #
    # class_index = 0
    # # shap.decision_plot(explainer.expected_value[class_index], shap_values[class_index], X_test.columns, return_objects=True)
    # shap.decision_plot(explainer.expected_value[class_index], shap_values[class_index], X_test.columns, return_objects=True, link="logit")

    """ Decision Plots Using Linguistic Features """
    # # X_train, X_test, y_train, y_test = load_isot_text_data()
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train[:20]
    # X_test = X_test[:20]
    # y_train = y_train[:20]
    # y_test = y_test[:20]
    #
    # X_train = X_train.drop(["text"], axis=1)
    # X_test = X_test.drop(["text"], axis=1)
    #
    # print("X_train Shape: {}".format(X_train.shape))
    # # X_train, X_test = perform_truncation(X_train, X_test, y_train, 100)
    # print("X_train Shape after TruncatedSVD: {}".format(X_train.shape))
    #
    # model = SVC(C=1, decision_function_shape='ovo', gamma=0.1, probability=True, random_state=42)
    # model.fit(X_train, y_train)
    #
    # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # shap_values = explainer.shap_values(X_test)
    #
    # # class_index = 4
    # # shap.decision_plot(explainer.expected_value[class_index], shap_values[class_index], X_test.columns, link="logit", feature_order="importance")
    # shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test.columns, link="logit",
    #                    feature_order="importance")
    # shap.decision_plot(explainer.expected_value[1], shap_values[1], X_test.columns, link="logit",
    #                    feature_order="importance")
    # shap.decision_plot(explainer.expected_value[2], shap_values[2], X_test.columns, link="logit",
    #                    feature_order="importance")
    # shap.decision_plot(explainer.expected_value[3], shap_values[3], X_test.columns, link="logit",
    #                    feature_order="importance")
    # shap.decision_plot(explainer.expected_value[4], shap_values[4], X_test.columns, link="logit",
    #                    feature_order="importance")
    # shap.decision_plot(explainer.expected_value[5], shap_values[5], X_test.columns, link="logit",
    #                    feature_order="importance")

    """ Produce SHAP plots for SVC Using TFIDF Features """
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train[:100]
    # X_test = X_test[:100]
    # y_train = y_train[:100]
    # y_test = y_test[:100]
    #
    # X_train, X_test, feature_names = create_tfidf_features(X_train, X_test, (1, 1))
    # X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
    # X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
    # print(X_train.shape)
    # # X_train, X_test = perform_truncation(X_train, X_test, y_train, 100)
    # print(X_train.shape)
    #
    # model = SVC(C=1, decision_function_shape='ovo', gamma=0.1, probability=True, random_state=42)
    # model.fit(X_train, y_train)
    #
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"\nAccuracy Score: {accuracy}")
    #
    # # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # # shap_values = explainer.shap_values(X_test)
    # # shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar")
    #
    # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values[0], X_test, feature_names=feature_names)
    # shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
    # shap.summary_plot(shap_values[2], X_test, feature_names=feature_names)
    # shap.summary_plot(shap_values[3], X_test, feature_names=feature_names)
    # shap.summary_plot(shap_values[4], X_test, feature_names=feature_names)
    # shap.summary_plot(shap_values[5], X_test, feature_names=feature_names)

    """ Produce SHAP plots for SVC Using Linguistic Features """
    # X_train, X_test, y_train, y_test = load_liar_data()
    # X_train = X_train[:100]
    # X_test = X_test[:100]
    # y_train = y_train[:100]
    # y_test = y_test[:100]
    #
    # X_train = X_train.drop(["text"], axis=1)
    # X_test = X_test.drop(["text"], axis=1)
    #
    # model = SVC(C=1, decision_function_shape='ovo', gamma=0.1, probability=True, random_state=42)
    # model.fit(X_train, y_train)
    #
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"\nAccuracy Score: {accuracy}")
    #
    # # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # # shap_values = explainer.shap_values(X_test)
    # # shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns, plot_type="bar")
    #
    # explainer = shap.KernelExplainer(model=model.predict_proba, data=X_train, link="logit")
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values[0], X_test)
    # shap.summary_plot(shap_values[1], X_test)
    # shap.summary_plot(shap_values[2], X_test)
    # shap.summary_plot(shap_values[3], X_test)
    # shap.summary_plot(shap_values[4], X_test)
    # shap.summary_plot(shap_values[5], X_test)


if __name__ == "__main__":
    main()
