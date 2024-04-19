import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def load_LIAR_data():
    LIAR_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 6]]
    y = LIAR_dataframe[LIAR_dataframe.columns[6]]

    return X, y


def get_linguistic_features():
    X, y = load_LIAR_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=42)

    X_train = X_train.drop("text", axis=1)
    X_test = X_test.drop("text", axis=1)

    return X_train, X_test, y_train, y_test


def get_tfidf_features():
    X, y = load_LIAR_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english', norm=None, ngram_range=(1, 1))
    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)

    X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    X_train_scaled_df, X_test_scaled_df = min_max_scale(X_train_df, X_test_df, y_train)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test


def min_max_scale(X_train_df, X_test_df, y_train):
    min_max_scaler = MinMaxScaler()

    scaled_X_train_data = min_max_scaler.fit_transform(X_train_df, y_train)
    scaled_X_train = pd.DataFrame(scaled_X_train_data, columns=X_train_df.columns)

    scaled_X_test_data = min_max_scaler.transform(X_test_df)
    scaled_X_test = pd.DataFrame(scaled_X_test_data, columns=X_test_df.columns)

    return scaled_X_train, scaled_X_test


def manual_classify(X_train, X_test, y_train, y_test):
    kernels = ['rbf', 'linear', 'sigmoid']

    for c in [0.1, 1, 10, 100]:
        for gamma in [0.1, 1, 10, 100, 1000, 10000]:
            dir_name = f"c{c}_g{gamma}_Linguistic_Ngrams1to1_2Classes_Lemma_Undersampled"
            os.mkdir(dir_name)
            for kernel in kernels:
                classifier = SVC(C=c, gamma=gamma, kernel=kernel, random_state=42)

                ovr_classifier = OneVsRestClassifier(classifier)
                ovr_classifier.fit(X_train, y_train)
                predicted_labels = ovr_classifier.predict(X_test)
                cm = confusion_matrix(y_test, predicted_labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ovr_classifier.classes_)
                disp.plot()
                report = classification_report(y_test, predicted_labels)
                print(f"C: {c}, Gamma: {gamma}, Kernel: {kernel}")
                print(f"OVR Classification Report: \n{report}")
                plt.title(f"Accuracy: {round(accuracy_score(y_test, predicted_labels) * 100, 2)}%")
                plt.savefig(f"{dir_name}\\Manual_TFIDF_c{c}_g{gamma}_{kernel}_ovr.png")

                classifier.fit(X_train, y_train)
                predicted_labels = classifier.predict(X_test)
                cm = confusion_matrix(y_test, predicted_labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
                disp.plot()
                report = classification_report(y_test, predicted_labels)
                print(f"C: {c}, Gamma: {gamma}, Kernel: {kernel}")
                print(f"OVO Classification Report: \n{report}")
                plt.title(f"Accuracy: {round(accuracy_score(y_test, predicted_labels) * 100, 2)}%")
                plt.savefig(f"{dir_name}\\Manual_TFIDF_c{c}_g{gamma}_{kernel}_ovo.png")


def grid_search(X_train, X_test, y_train, y_test):
    """
    A high value of C is like having a very fine, sharp stick that you're willing to wiggle around every marble to
    avoid touching any of them — this represents a model with low bias but potentially high variance (overfitting).

    Conversely, a low value of C is akin to using a broad, dull stick, where you draw a thick line that's okay with
    covering some marbles in the process — indicating a model that's not too strict about misclassifications and thus
    may have high bias but lower variance (underfitting).

    Think of gamma as the gravitational force exerted by each marble on the line you're drawing in the sand.
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 1, 10, 100, 1000],
                  'kernel': ['rbf', 'sigmoid', 'linear'], 'decision_function_shape': ['ovo', 'ovr']}

    grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=10, scoring='f1_macro', n_jobs=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    cm = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
    disp.plot()
    plt.title(f"Accuracy: {round(accuracy_score(y_test, grid_predictions) * 100, 2)}%")
    plt.savefig("TFIDF_PCA100_f1macro_ngrams1to1.png")
    report = classification_report(y_test, grid_predictions)
    print(f"Classification Report: {report}\n")


def perform_pca(X_train, X_test, y_train, num_components):
    pca = PCA(n_components=num_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train, y_train)
    X_test_pca = pca.transform(X_test)

    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca.get_feature_names_out())
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca.get_feature_names_out())

    X_train_pca, X_test_pca = min_max_scale(X_train_pca_df, X_test_pca_df, y_train)

    return X_train_pca, X_test_pca


def explain_test_set(X_train, X_test, y_train, y_test):
    classifier = SVC(C=100, class_weight='balanced', decision_function_shape='ovo', gamma=0.01, random_state=42, probability=True)
    classifier.fit(X_train, y_train)

    explainer = shap.KernelExplainer(classifier.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)


def main():
    # X_train_linguistic, X_test_linguistic, y_train_linguistic, y_test_linguistic = get_linguistic_features()
    # X_train, X_test, y_train, y_test = X_train_linguistic, X_test_linguistic, y_train_linguistic, y_test_linguistic

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = get_tfidf_features()
    X_train, X_test, y_train, y_test = X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf
    X_train.plot()

    # # X_train, X_test = perform_pca(X_train, X_test, y_train, num_components=500)

    # X_train_linguistic = X_train_linguistic.reset_index(drop=True)
    # X_test_linguistic = X_test_linguistic.reset_index(drop=True)
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    # X_train = X_train.join(X_train_linguistic)
    # X_test = X_test.join(X_test_linguistic)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # # """New class distribution"""
    # y_train.replace([0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 1, 1], inplace=True)
    # y_test.replace([0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 1, 1], inplace=True)

    """ !DO NOT RESAMPLE THE TEST SET OR VALIDATION SET! 
        You should make your own Cross-Validation for Grid Search!
    """
    # under_sampler = RandomUnderSampler(random_state=42)
    # X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    # over_sampler = SMOTE(random_state=42)
    # X_train, y_train = over_sampler.fit_resample(X_train, y_train)

    # X_train, X_test = perform_pca(X_train, X_test, y_train, num_components=10)

    print(f"X_train: {X_train}\n")
    print(f"X_test: {X_test}\n")
    print(f"Train Label Distribution: {y_train.value_counts()}\n")
    print(f"Test Label Distribution: {y_test.value_counts()}\n")

    # grid_search(X_train, X_test, y_train, y_test)
    # manual_classify(X_train, X_test, y_train, y_test)

    explain_test_set(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
