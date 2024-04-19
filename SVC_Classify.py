import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
import joblib


def load_liar_data():
    LIAR_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 22]]
    y = LIAR_dataframe[LIAR_dataframe.columns[22]]

    return X, y


def load_isot_data():
    ISOT_dataframe = pd.read_csv("processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT.csv")
    ISOT_dataframe = ISOT_dataframe.dropna()

    X = ISOT_dataframe[ISOT_dataframe.columns[0: 22]]
    y = ISOT_dataframe[ISOT_dataframe.columns[22]]

    return X, y


def create_tfidf_features(X_train, X_test, ngram_range):
    tfidf = TfidfVectorizer(ngram_range=ngram_range)
    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)

    return X_train_tfidf, X_test_tfidf


def get_linguistic_features(X_train, X_test):
    X_train = X_train.drop("text", axis=1)
    X_test = X_test.drop("text", axis=1)

    return X_train, X_test


def min_max_scale(X_train, X_test, y_train):
    min_max_scaler = MaxAbsScaler()

    scaled_X_train = min_max_scaler.fit_transform(X_train, y_train)
    scaled_X_test = min_max_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def grid_search_svc(X_train, X_test, y_train, y_test, ngram_range, scoring_metrics, param_grid):
    for scoring_metric in scoring_metrics:
        grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, refit=True, verbose=10, scoring=scoring_metric, n_jobs=8)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        cm = confusion_matrix(y_test, grid_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
        disp.plot()
        accuracy = round(accuracy_score(y_test, grid_predictions) * 100, 2)
        plt.title(f"Accuracy Score: {accuracy}%")
        plt.suptitle(f"Model Params: [{grid.best_estimator_}]", fontsize="small")
        plt.savefig(f"Linguistic_{scoring_metric}_ngrams{ngram_range}_ISOT_Undersampled.png")
        report = classification_report(y_test, grid_predictions)
        print(f"Classification Report: {report}\n")
        joblib.dump(grid, f"best_model_{scoring_metric}_Linguistic_ISOT_saved_accuracy_{accuracy}_Undersampled.pkl")


def undersample(X_train, y_train):
    under_sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def re_label(y_train, y_test, new_class_labels):
    y_train.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)
    y_test.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)

    return y_train, y_test


def perform_truncation(X_train, X_test, y_train, num_components):
    pca = TruncatedSVD(n_components=num_components, n_iter=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train, y_train)
    X_test_pca = pca.transform(X_test)

    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca.get_feature_names_out())
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca.get_feature_names_out())

    X_train_pca, X_test_pca = min_max_scale(X_train_pca_df, X_test_pca_df, y_train)

    return X_train_pca, X_test_pca


def main():
    test_size = 0.3
    validation_size = 0.5
    ngram_range = (1, 1)
    scoring_metrics = ["accuracy"]
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [0.1, 1, 10, 100, 1000],
                  'kernel': ['rbf', 'sigmoid', 'linear']}
    new_class_labels = [0, 0, 1, 1, 1, 1]

    # X, y = load_liar_data()
    X, y = load_isot_data()

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=validation_size, stratify=y, shuffle=True, random_state=42)

    # y_train, y_test = re_label(y_train, y_test, new_class_labels)
    # X_train, y_train = undersample(X_train, y_train)

    # X_train, X_test = create_tfidf_features(X_train, X_test, ngram_range)
    # X_train, X_test = min_max_scale(X_train, X_test, y_train)

    X_train, X_test = get_linguistic_features(X_train, X_test)
    # X_train = X_train[['adverb_count', 'linsear_write_formula', 'monosyllable_count',
    #                    'conjunction_count', 'pronoun_count']]
    # X_test = X_test[['adverb_count', 'linsear_write_formula', 'monosyllable_count',
    #                  'conjunction_count', 'pronoun_count']]

    X_train, y_train = undersample(X_train, y_train)
    # X_train, X_test = perform_pca(X_train, X_test, y_train, num_components=100)

    print(f"Train Label Distribution: {y_train.value_counts()}\n")
    print(f"Test Label Distribution: {y_test.value_counts()}\n")
    print(f"X Train Shape: {X_train.shape}")
    print(f"X Test Shape: {X_test.shape}\n")

    grid_search_svc(X_train, X_test, y_train, y_test, ngram_range, scoring_metrics, param_grid)


if __name__ == "__main__":
    main()
