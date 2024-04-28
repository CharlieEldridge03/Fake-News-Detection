import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import joblib
from sklearn.model_selection import StratifiedKFold


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


def get_linguistic_features(X_train, X_test):
    X_train = X_train.drop("text", axis=1)
    X_test = X_test.drop("text", axis=1)

    # Re-scale the combined train and validation values
    scaled_X_train, scaled_X_test = min_max_scale(X_train, X_test)

    return scaled_X_train, scaled_X_test


def undersample(X_train, y_train):
    under_sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def re_label(y_train, y_test, new_class_labels):
    y_train.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)
    y_test.replace([0, 1, 2, 3, 4, 5], new_class_labels, inplace=True)

    return y_train, y_test


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


def grid_search_svc(X_train, y_train, X_test, y_test, ngram_range, scoring_metrics, param_grid):
    for scoring_metric in scoring_metrics:
        k_folds = StratifiedKFold(n_splits=5)
        grid = GridSearchCV(SVC(random_state=42), param_grid, cv=k_folds, refit=True, verbose=10,
                            scoring=scoring_metric, n_jobs=8)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        cm = confusion_matrix(y_test, grid_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
        disp.plot()
        accuracy = round(accuracy_score(y_test, grid_predictions) * 100, 2)
        plt.title("Accuracy Score: {}%".format(accuracy))
        plt.suptitle("Model Params: [{}]".format(grid.best_estimator_), fontsize="small")
        plt.savefig("ISOT_Text_Trunc100_{}_ngrams{}_TFIDF_Lemma.png".format(scoring_metric, ngram_range))
        report = classification_report(y_test, grid_predictions)
        print("Classification Report: {}\n".format(report))
        joblib.dump(grid, "ISOT_Text_Trunc100_{}_ngrams{}_TFIDF_Lemma.pkl".format(scoring_metric, ngram_range))


def main():
    ngram_range = (2, 2)
    scoring_metric = ["accuracy", "f1_macro"]
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.01, 0.1, 1, 10, 100, 1000],
                  'kernel': ['rbf', 'sigmoid', 'linear']}

    X_train, X_test, y_train, y_test = load_isot_title_data()

    # RELABEL LIAR DATASET TO USE 2 CLASSES
    # new_class_labels = [0, 0, 1, 1, 1, 1]
    # y_train, y_test = re_label(y_train, y_test, new_class_labels)

    # TFIDF FEATURES
    X_train, X_test = create_tfidf_features(X_train, X_test, ngram_range)

    # LINGUISTIC FEATURES
    # X_train, X_test = get_linguistic_features(X_train, X_test)

    # UNDERSAMPLE
    # X_train, y_train = undersample(X_train, y_train)

    # TRUNCATE
    X_train, X_test = perform_truncation(X_train, X_test, y_train, 100)

    # OUTPUT LABEL DISTRIBUTIONS
    print("Train Label Distribution: {}\n".format(y_train.value_counts()))
    print("Test Label Distribution: {}\n".format(y_test.value_counts()))

    grid_search_svc(X_train, y_train, X_test, y_test, ngram_range, scoring_metric, param_grid)


if __name__ == "__main__":
    main()

