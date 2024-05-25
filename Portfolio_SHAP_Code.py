"""

"""

import sklearn
import imblearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import joblib
import shap
import os


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test


def min_max_scale(X_train, X_test):
    # MaxAbsScalar is used for compatibility with Sparce Matrices.
    min_max_scaler = MaxAbsScaler()

    scaled_X_train = min_max_scaler.fit_transform(X_train)
    scaled_X_test = min_max_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def create_tfidf_features(X_train, X_test, ngram_range):
    # Ensure that TFIDF values appear in a minimum of 1% of documents to capture only "important" features.
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=0.01)

    X_train_tfidf = tfidf.fit_transform(X_train.text)
    X_test_tfidf = tfidf.transform(X_test.text)
    feature_names = tfidf.get_feature_names_out()

    # The newly extracted TFIDF features are scaled from 0 to 1.
    scaled_X_train, scaled_X_test = min_max_scale(X_train_tfidf, X_test_tfidf)

    return scaled_X_train, scaled_X_test, feature_names


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


def create_ensemble(X_train, y_train, linear_svc, rbf_svc, sigmoid_svc, ensemble_file_name):
    # If an ensemble model already exists with the specified name, load that model and return it.
    if os.path.exists(ensemble_file_name + ".pkl"):
        print("Existing ensemble model has been found and will be loaded.")
        ensemble_svc = joblib.load(ensemble_file_name + ".pkl")

    # Otherwise, create a new ensemble model using the given models, trained on the given data, and output with the
    # specified file name.
    else:
        # Setting n_jobs to a larger value or -1 will likely speed up the training process.
        # This value is set to 1 for "scipy" library compatibility reasons.
        ensemble_svc = StackingClassifier(
            estimators=[('sigmoid_svc', sigmoid_svc), ('rbf_svc', rbf_svc)], final_estimator=linear_svc,
            verbose=1, n_jobs=-1)

        ensemble_svc.fit(X_train, y_train)
        joblib.dump(ensemble_svc, ensemble_file_name + ".pkl")

    return ensemble_svc


def load_data(dataset_name):
    if dataset_name == "liar_dataset":
        X_train, X_test, y_train, y_test = load_liar_data()

    elif dataset_name == "isot_titles_dataset":
        X_train, X_test, y_train, y_test = load_isot_title_data()

    elif dataset_name == "isot_texts_dataset":
        X_train, X_test, y_train, y_test = load_isot_text_data()

    else:
        X_train, X_test, y_train, y_test = load_liar_data()

    return X_train, X_test, y_train, y_test


def get_feature_set(X_train, X_test, feature_set_name):
    if feature_set_name == "tfidf":
        X_train, X_test, feature_names = create_tfidf_features(X_train, X_test, (1, 1))
    else:
        X_train, X_test = get_linguistic_features(X_train, X_test)
        feature_names = X_train.columns

    return X_train, X_test, feature_names


def load_optimal_base_models(dataset_name):
    if dataset_name == "liar_dataset":
        linear_svc = SVC(C=1000, gamma=0.01, kernel="linear", probability=True, random_state=42, verbose=1)
        rbf_svc = SVC(C=100, gamma=0.1, kernel="rbf", probability=True, random_state=42, verbose=1)
        sigmoid_svc = SVC(C=1000, gamma=0.01, kernel="sigmoid", probability=True, random_state=42, verbose=1)

    elif dataset_name == "isot_titles_dataset":
        linear_svc = SVC(C=100, gamma=0.01, kernel="linear", probability=True, random_state=42, verbose=1)
        rbf_svc = SVC(C=10, gamma=1, kernel="rbf", probability=True, random_state=42, verbose=1)
        sigmoid_svc = SVC(C=1000, gamma=0.01, kernel="sigmoid", probability=True, random_state=42, verbose=1)

    elif dataset_name == "isot_texts_dataset":
        linear_svc = SVC(C=1000, gamma=0.01, kernel="linear", probability=True, random_state=42, verbose=1)
        rbf_svc = SVC(C=10, gamma=1, kernel="rbf", probability=True, random_state=42, verbose=1)
        sigmoid_svc = SVC(C=1000, gamma=0.01, kernel="sigmoid", probability=True, random_state=42, verbose=1)

    else:
        linear_svc = SVC(kernel="linear", probability=True, random_state=42, verbose=1)
        rbf_svc = SVC(kernel="rbf", probability=True, random_state=42, verbose=1)
        sigmoid_svc = SVC(kernel="sigmoid", probability=True, random_state=42, verbose=1)

    return linear_svc, rbf_svc, sigmoid_svc


def display_summary_plots(shap_values, X_test, feature_names, dataset_name, ensemble_file_name):
    if dataset_name == "liar_dataset":
        num_classes = 6
    else:
        num_classes = 2

    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      title="Feature Importance Summary for the {}".format(dataset_name), show=False,
                      matplotlib=True).savefig("{}_summary_bar_plot.png".format(ensemble_file_name))

    for i in range(0, num_classes):
        shap.summary_plot(shap_values[i], X_test, feature_names=feature_names, plot_type="bar",
                          title="{} Shapley Values Summary for Class {}".format(dataset_name, i), show=False,
                          matplotlib=True).savefig("{}_class{}_bar_plot.png".format(ensemble_file_name, i))
        shap.summary_plot(shap_values[i], X_test, feature_names=feature_names,
                          title="{} Shapley Values Feature Impacts for Class {}".format(dataset_name, i), show=False,
                          matplotlib=True).savefig("{}_class{}_scatter_plot.png".format(ensemble_file_name, i))


def perform_experiment(dataset_name, feature_set_name):
    # This will be used as the base file name for any ensemble model files and confusion matrices produced.
    ensemble_file_name = "{}_{}_StackingEnsemble".format(dataset_name, feature_set_name)

    if dataset_name == "liar_dataset" and feature_set_name == "linguistic":
        print("There is no experiment currently setup for this configuration.\nPlease try another experiment.")
        exit()

    # Loading the target dataset data.
    X_train, X_test, y_train, y_test = load_data(dataset_name)
    # Obtaining the target feature set.
    X_train, X_test, feature_names = get_feature_set(X_train, X_test, feature_set_name)

    print("Ensemble is being trained on data with the following shape: {}".format(X_train.shape))

    if dataset_name == "liar_dataset" and feature_set_name == "tfidf":
        X_train, y_train = oversample(X_train, y_train)

    linear_svc, rbf_svc, sigmoid_svc = load_optimal_base_models(dataset_name)
    ensemble = create_ensemble(X_train, y_train, linear_svc, rbf_svc, sigmoid_svc, ensemble_file_name)

    y_pred = ensemble.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    print("Accuracy: {}%".format(accuracy))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble.classes_)
    disp.plot()
    plt.title("Accuracy Score: {}%".format(accuracy))
    plt.savefig("{}_ConfusionMatrix.png".format(ensemble_file_name))
    plt.clf()

    if feature_set_name == "tfidf":
        if dataset_name == "liar_dataset":
            X_test = pd.DataFrame(X_test.toarray(), columns=feature_names).sample(n=1000, random_state=42)
            background_data = pd.DataFrame(X_train.toarray(), columns=feature_names).sample(n=100, random_state=42)

        else:
            X_test = pd.DataFrame(X_test.toarray(), columns=feature_names).sample(n=10, random_state=42)
            background_data = pd.DataFrame(X_train.toarray(), columns=feature_names).sample(n=10, random_state=42)

    else:
        X_test = pd.DataFrame(X_test.toarray(), columns=feature_names).sample(n=1000, random_state=42)
        background_data = pd.DataFrame(X_train.toarray(), columns=feature_names).sample(n=500, random_state=42)

    print("Calculating SHAP values with data of shape: {}".format(X_train.shape))
    explainer = shap.KernelExplainer(model=ensemble.predict_proba, data=background_data,
                                     feature_names=background_data.columns, link="logit")
    shap_values = explainer.shap_values(X_test)
    display_summary_plots(shap_values, X_test, feature_names, dataset_name, ensemble_file_name)


def main():
    """
    SHAP Experiment Details
    ------------------------
    TruncatedSVD is not used here because it is important for the relationship between the TFIDF values and feature
    names to be preserved for the SHAP experiments.

    These experiments can be very computationally expensive and usually take hours to run even when using small subsets
    of the original data. For this reason the ensemble model files featured in the report have likely been included and
    will be loaded automatically to save time otherwise spent training the ensemble models.

    Experiment Notes:
        * The TFIDF experiments in this file only use TFIDF values for features that appear in a minimum of 1% of each
            dataset corpus.

        * SHAP values will be generated using a randomly selected subset of the original datasets to save computation
            time. The subset sizes vary between each experiment (due to differences in the number of features used)
            and are as follows:

                LIAR Dataset:
                    TFIDF Experiment - [100 test cases, 50 background inference samples]

                ISOT Datasets:
                    TFIDF Experiments - [10 test cases, 10 background inference samples]
                    Linguistic Experiments - [100 test cases, 50 background inference samples]
    """

    # Specifying which dataset to use [Options are: "liar_dataset", "isot_titles_dataset", "isot_texts_dataset"]
    dataset_name = "liar_dataset"
    # Specifying which feature set to use [Options are: "tfidf", "linguistic"]
    feature_set_name = "tfidf"

    perform_experiment(dataset_name, feature_set_name)


if __name__ == "__main__":
    main()
