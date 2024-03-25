import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def load_LIAR_data():
    LIAR_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 6]]
    y = LIAR_dataframe[LIAR_dataframe.columns[6]]

    return X, y


def get_tfidf_dataframes(X_train_text, X_test_text):
    tfidf = TfidfVectorizer(stop_words='english', norm=None)

    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    return X_train_df, X_test_df


def get_linguistic_features():
    X, y = load_LIAR_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=42)

    X_train = X_train.drop("text", axis=1)
    X_test = X_test.drop("text", axis=1)

    return X_train, X_test, y_train, y_test


def get_tfidf_features():
    X, y = load_LIAR_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english', norm=None)
    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)

    X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    X_train_scaled_df, X_test_scaled_df = min_max_scale(X_train_df, X_test_df)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test


def min_max_scale(X_train_df, X_test_df):
    min_max_scaler = MinMaxScaler()

    scaled_X_train_data = min_max_scaler.fit_transform(X_train_df)
    scaled_X_train = pd.DataFrame(scaled_X_train_data, columns=X_train_df.columns)

    scaled_X_test_data = min_max_scaler.transform(X_test_df)
    scaled_X_test = pd.DataFrame(scaled_X_test_data, columns=X_test_df.columns)

    return scaled_X_train, scaled_X_test


def classify(X_train, X_test, y_train, y_test):
    param_grid = {'C': [0.1, 1, 1000], 'gamma': [0.01, 1, 100],
                  'kernel': ['rbf', 'linear', 'sigmoid']}

    grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=10, scoring='f1_macro', n_jobs=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    cm = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
    disp.plot()
    plt.savefig("CM_TFIDF_PCA5_f1_macro.png")
    report = classification_report(y_test, grid_predictions)
    print(f"Classification Report: {report}\n")
    print(f"Accuracy Score: {accuracy_score(y_test, grid_predictions) * 100}%")


def perform_pca(X_train, X_test, y_train, num_components):
    pca = PCA(n_components=num_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train, y_train)
    X_test_pca = pca.transform(X_test)

    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca.get_feature_names_out())
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca.get_feature_names_out())

    X_train_pca, X_test_pca = min_max_scale(X_train_pca_df, X_test_pca_df)

    return X_train_pca, X_test_pca


def main():
    # X_train_linguistic, X_test_linguistic, y_train_linguistic, y_test_linguistic = get_linguistic_features()
    # X_train, X_test, y_train, y_test = X_train_linguistic, X_test_linguistic, y_train_linguistic, y_test_linguistic

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = get_tfidf_features()
    X_train, X_test, y_train, y_test = X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf
    X_train, X_test = perform_pca(X_train, X_test, y_train, num_components=5)

    # X_train_linguistic = X_train_linguistic.reset_index(drop=True)
    # X_test_linguistic = X_test_linguistic.reset_index(drop=True)
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    # X_train = X_train.join(X_train_linguistic)
    # X_test = X_test.join(X_test_linguistic)

    print(f"X_train: {X_train}\n")
    print(f"X_test: {X_test}\n")
    print(f"Train Label Distribution: {y_train.value_counts()}\n")
    print(f"Test Label Distribution: {y_test.value_counts()}\n")

    classify(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
