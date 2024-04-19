import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns


def load_liar_data():
    LIAR_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 6]]
    y = LIAR_dataframe[LIAR_dataframe.columns[6]]

    return X, y


def create_train_test_split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train, X_test, ngram_range):
    tfidf = TfidfVectorizer(norm='l1', ngram_range=ngram_range)
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
        grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, refit=True, verbose=10, scoring=scoring_metric,
                            n_jobs=8)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        cm = confusion_matrix(y_test, grid_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
        disp.plot()
        plt.title(f"Accuracy Score: {round(accuracy_score(y_test, grid_predictions) * 100, 2)}%")
        plt.suptitle(f"Model Params: [{grid.best_estimator_}]", fontsize="small")
        plt.savefig(f"Linguistic_{scoring_metric}_ngrams{ngram_range}_2Classes_2True4False.png")
        report = classification_report(y_test, grid_predictions)
        print(f"Classification Report: {report}\n")


def undersample(X_train, y_train):
    under_sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def re_label(y_train, y_test, new_class_labels):
    y_train.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)
    y_test.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)

    return y_train, y_test


def k_fold(X_train, y_train, param_grid):
    kfold = KFold(n_splits=5, shuffle=False)

    for key in param_grid:
        print(key, param_grid[key])

    # classifier = SVC(random_state=42)
    # accuracies = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=kfold)


def main():
    test_size = 0.3
    ngram_range = (1, 1)
    scoring_metrics = ["f1_macro", "accuracy"]
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'kernel': ['rbf', 'sigmoid', 'linear'],
                  'class_weight': ['balanced', None],
                  'decision_function_shape': ['ovo', 'ovr']}

    X, y = load_liar_data()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size)
    X_train, X_test = create_tfidf_features(X_train, X_test, ngram_range)
    X_train, X_test = min_max_scale(X_train, X_test, y_train)

    print(f"Train Label Distribution: {y_train.value_counts()}\n")
    print(f"Test Label Distribution: {y_test.value_counts()}\n")
    print(f"X Train Shape: {X_train.shape}")
    print(f"X Test Shape: {X_test.shape}\n")

    # k_fold(X_train, y_train, param_grid)
    # grid_search_svc(X_train, X_test, y_train, y_test, ngram_range, scoring_metrics, param_grid)


if __name__ == "__main__":
    main()
