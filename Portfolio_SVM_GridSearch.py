"""
Portfolio_SVM_GridSearch.py: Performs a grid search to find the optimal hyperparameter values of an SVM model.

Author: Charlie Eldridge
Last Updated: 03/06/2024
Version: "2.0"
email: charlieeldridge03@gmail.com
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd


def load_liar_data():
    LIAR_train_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_train.csv",
                                       index_col=False).dropna()
    LIAR_test_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_test.csv",
                                      index_col=False).dropna()
    LIAR_valid_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_valid.csv",
                                       index_col=False).dropna()

    # Separate the labels from the rest of the training data.
    X_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[0: 22]]
    y_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[22]]

    # Separate the labels from the rest of the test data.
    X_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[0: 22]]
    y_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[22]]

    # Separate the labels from the rest of the validation data.
    X_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[0: 22]]
    y_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[22]]

    # Combining the test/validation sets to create an approximately 20% train/test split.
    X_test = pd.concat([X_test, X_valid], ignore_index=True)
    y_test = pd.concat([y_test, y_valid], ignore_index=True)

    return X_train, X_test, y_train, y_test


def load_isot_text_data():
    # Read in the preprocessed ISOT article text data, ignoring the index column.
    ISOT_text_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_texts.csv",
        index_col=False).dropna()

    # Separate the labels from the rest of the training data.
    X = ISOT_text_dataframe[ISOT_text_dataframe.columns[0: 22]]
    y = ISOT_text_dataframe[ISOT_text_dataframe.columns[22]]

    # 20% Train/Test Split to reflect split used for the LIAR dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def load_isot_title_data():
    # Read in the preprocessed ISOT titles data, ignoring the index column.
    ISOT_title_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_titles.csv",
        index_col=False).dropna()

    # Separate the labels from the rest of the training data.
    X = ISOT_title_dataframe[ISOT_title_dataframe.columns[0: 22]]
    y = ISOT_title_dataframe[ISOT_title_dataframe.columns[22]]

    # 20% Train/Test Split to reflect split used for the LIAR dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def min_max_scale(X_train, X_test):
    # MaxAbsScalar is used for compatibility with Sparce Matrices.
    min_max_scaler = MaxAbsScaler()

    scaled_X_train = min_max_scaler.fit_transform(X_train)
    scaled_X_test = min_max_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def create_tfidf_features(X_train, X_test, ngram_range, use_min_df):
    if use_min_df:
        min_df = 0.01
    else:
        min_df = 1

    # Extract TFIDF values using the given ngram range.
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)

    # Transform both train and test sets using the features found in the training data preserve dimensionality.
    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)

    # The newly extracted TFIDF features are scaled from 0 to 1.
    scaled_X_train, scaled_X_test = min_max_scale(X_train_tfidf, X_test_tfidf)

    return scaled_X_train, scaled_X_test


def get_linguistic_features(X_train, X_test):
    # The text column can be discarded when using the linguistic features.
    X_train = X_train.drop("text", axis=1)
    X_test = X_test.drop("text", axis=1)

    return X_train, X_test


def undersample(X_train, y_train):
    under_sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def oversample(X_train, y_train):
    over_sampler = SMOTE(random_state=42)
    X_train, y_train = over_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def load_data(dataset_name):
    if dataset_name == "liar_dataset":
        print("Loading LIAR Dataset.")
        X_train, X_test, y_train, y_test = load_liar_data()

    elif dataset_name == "isot_titles_dataset":
        print("Loading ISOT Titles Dataset.")
        X_train, X_test, y_train, y_test = load_isot_title_data()

    elif dataset_name == "isot_texts_dataset":
        print("Loading ISOT Texts Dataset.")
        X_train, X_test, y_train, y_test = load_isot_text_data()

    else:
        X_train, X_test, y_train, y_test = load_liar_data()

    return X_train, X_test, y_train, y_test


def get_feature_set(X_train, X_test, feature_set_name, ngram_range, use_min_df):
    if feature_set_name == "tfidf":
        print("Using TF-IDF Feature Set.")
        X_train, X_test = create_tfidf_features(X_train, X_test, ngram_range, use_min_df)
    else:
        print("Using Linguistic Feature Set.")
        X_train, X_test = get_linguistic_features(X_train, X_test)

    return X_train, X_test


def perform_svd(X_train, X_test, y_train, num_components):
    # Truncated SVD is used for compatibility with sparse matrices to reduce the dimensionality of the TFIDF features
    # used for training models on the ISOT dataset.
    svd = TruncatedSVD(n_components=num_components, n_iter=5, random_state=42)
    X_train_svd = svd.fit_transform(X_train, y_train)
    X_test_svd = svd.transform(X_test)

    X_train_svd_df = pd.DataFrame(X_train_svd, columns=svd.get_feature_names_out())
    X_test_svd_df = pd.DataFrame(X_test_svd, columns=svd.get_feature_names_out())

    # The newly transformed TFIDF values are re-scaled back into the range 0 to 1.
    min_max_scaler = MinMaxScaler()
    scaled_X_train = min_max_scaler.fit_transform(X_train_svd_df)
    scaled_X_test = min_max_scaler.transform(X_test_svd_df)

    return scaled_X_train, scaled_X_test


def grid_search_svc(X_train, y_train, X_test, y_test, ngram_range, scoring_metrics, param_grid):
    # For each of the scoring metrics defined (i.e. "accuracy" and "f1-macro") perform a cross validated grid search
    # to find the model that achieves the highest score for that metric.
    for scoring_metric in scoring_metrics:
        # Grid search is performed using a stratified 5-fold cross validation set to reflect the base class distribution.
        k_folds = StratifiedKFold(n_splits=5)
        base_model = SVC(random_state=42)
        grid = GridSearchCV(base_model, param_grid, cv=k_folds, refit=True, verbose=10, scoring=scoring_metric, n_jobs=-1)
        grid.fit(X_train, y_train)

        # Output the optimal configuration found using cross-validated grid search.
        print(grid.best_estimator_)

        # Output and save the confusion matrix produced by the optimal model when evaluated on the test set.
        grid_predictions = grid.predict(X_test)
        cm = confusion_matrix(y_test, grid_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
        disp.plot()
        accuracy = round(accuracy_score(y_test, grid_predictions) * 100, 2)
        plt.title("Accuracy Score: {}%".format(accuracy))
        plt.suptitle("Model Params: [{}]".format(grid.best_estimator_), fontsize="small")
        plt.savefig("ISOT_Text_Linguistic_EnsemblePrep_Linear_{}_ngrams{}.png".format(scoring_metric, ngram_range))

        # Output the classification report of the model to assess learning and performance on test set.
        report = classification_report(y_test, grid_predictions)
        print("Classification Report: {}\n".format(report))


def perform_experiment(ngram_range, scoring_metrics, param_grid, dataset_name, feature_set_name,
                       feature_set_modification, use_min_df):
    # Loading the target dataset data.
    X_train, X_test, y_train, y_test = load_data(dataset_name)
    # Obtaining the target feature set.
    X_train, X_test = get_feature_set(X_train, X_test, feature_set_name, ngram_range, use_min_df)

    if "isot" in dataset_name:
        X_train, X_test = perform_svd(X_train, X_test, y_train, 100)

    if feature_set_modification == "undersample":
        X_train, y_train = undersample(X_train, y_train)
    elif feature_set_modification == "oversample":
        X_train, y_train = oversample(X_train, y_train)

    # Output the dimensionality of the training and test data for manual sanity checks.
    print("Training Data Shape: {}".format(X_train.shape))
    print("Test Data Shape: {}\n".format(X_test.shape))

    # Display the label distributions to check the class distributions.
    print("Train Label Distribution: {}\n".format(y_train.value_counts()))
    print("Test Label Distribution: {}\n".format(y_test.value_counts()))

    grid_search_svc(X_train, y_train, X_test, y_test, ngram_range, scoring_metrics, param_grid)


def main():
    # Defining the grid-search parameters.
    ngram_range = (1, 1)
    scoring_metrics = ["accuracy", "f1_macro"]
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.01, 0.1, 1, 10, 100, 1000],
                  'kernel': ['linear', 'rbf', 'sigmoid']}
    # Specifying which dataset to use [Options are: "liar_dataset", "isot_titles_dataset", "isot_texts_dataset"]
    dataset_name = "liar_dataset"
    # Specifying which feature set to use [Options are: "tfidf", "linguistic"]
    feature_set_name = "tfidf"
    # Specifying how to modify the feature set [Options are: "undersample", "oversample", "none"]
    feature_set_modification = "oversample"
    # Specifying whether to only use TF-IDF features that appear in a minimum of 1% of documents
    # [Options are: "True", "False"]
    use_min_df = True

    perform_experiment(ngram_range, scoring_metrics, param_grid, dataset_name, feature_set_name,
                       feature_set_modification, use_min_df)


if __name__ == "__main__":
    main()
