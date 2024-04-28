import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
import matplotlib.pyplot as plt


def load_liar_data():
    LIAR_dataframe = pd.read_csv("processed_datasets_nonLemma/LIAR_Dataset_processed/processed_LIAR.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 22]]
    y = LIAR_dataframe[LIAR_dataframe.columns[22]]

    return X, y


def load_isot_data():
    LIAR_dataframe = pd.read_csv("processed_datasets_nonLemma/ISOT_Fake_News_Dataset_processed/processed_ISOT.csv")
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0: 22]]
    y = LIAR_dataframe[LIAR_dataframe.columns[22]]

    return X, y


def min_max_scale(X_train, X_test, y_train):
    min_max_scaler = MinMaxScaler()

    scaled_X_train = min_max_scaler.fit_transform(X_train, y_train)
    scaled_X_test = min_max_scaler.transform(X_test)

    return scaled_X_train, scaled_X_test


def main():
    print(f"Feature Importances for LIAR Dataset:\n")
    X, y = load_liar_data()
    X = X.drop("text", axis=1)
    chi2_selector = SelectKBest(chi2, k=3)
    chi2_selector.fit(X, y)
    selected_features = X.columns[chi2_selector.get_support()]
    print("Selected Chi Square Features:")
    print(selected_features, "\n")

    full_train = X.join(y).dropna()
    true_df = full_train[full_train["label"] == 0].drop(["label"], axis=1)
    mostly_true_df = full_train[full_train["label"] == 1].drop(["label"], axis=1)
    half_true_df = full_train[full_train["label"] == 2].drop(["label"], axis=1)
    barely_true_df = full_train[full_train["label"] == 3].drop(["label"], axis=1)
    false_df = full_train[full_train["label"] == 4].drop(["label"], axis=1)
    pants_fire_df = full_train[full_train["label"] == 5].drop(["label"], axis=1)

    df_array = [["r", true_df], ["b", mostly_true_df], ["g", half_true_df], ["c", barely_true_df], ["m", false_df],
                ["y", pants_fire_df]]

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(projection='3d')
    #
    # for tuple in df_array:
    #     colour = tuple[0]
    #     df = tuple[1]
    #
    #     ax.scatter(df['adjective_count'], df['lexicon_count'], df['conjunction_count'], c=colour, label=df_array.index(tuple))
    #
    # ax.set_xlabel('adjective_count')
    # ax.set_ylabel('lexicon_count')
    # ax.set_zlabel('conjunction_count')
    # plt.legend()
    # plt.show()

    # for tuple in df_array:
    #     colour = tuple[0]
    #     df = tuple[1]
    #
    #     plt.scatter(df['adjective_count'], df['lexicon_count'], c=colour, label=df_array.index(tuple))
    #
    # plt.xlabel('adjective_count')
    # plt.ylabel('lexicon_count')
    # plt.legend()
    # plt.show()


    print(f"\nFeature Importances for ISOT Dataset:\n")
    X, y = load_isot_data()
    X = X.drop("text", axis=1)
    chi2_selector = SelectKBest(chi2, k=10)
    chi2_selector.fit(X, y)
    selected_features = X.columns[chi2_selector.get_support()]
    print("Selected Chi Square Features:")
    print(selected_features, "\n")

    full_train = X.join(y).dropna()
    true_df = full_train[full_train["label"] == 0].drop(["label"], axis=1)
    false_df = full_train[full_train["label"] == 1].drop(["label"], axis=1)

    # df_array = [["b", true_df], ["r", false_df]]
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(projection='3d')
    #
    # for tuple in df_array:
    #     colour = tuple[0]
    #     df = tuple[1]
    #
    #     ax.scatter(df['adverb_count'], df['pronoun_count'], df['present_tense_count'], c=colour, label=df_array.index(tuple))
    #
    # ax.set_xlabel('adverb_count')
    # ax.set_ylabel('pronoun_count')
    # ax.set_zlabel('present_tense_count')
    # plt.legend()
    # plt.show()

    # for tuple in df_array:
    #     colour = tuple[0]
    #     df = tuple[1]
    #
    #     plt.scatter(df['adverb_count'], df['pronoun_count'], c=colour, label=df_array.index(tuple))
    #
    # plt.xlabel('adverb_count')
    # plt.ylabel('linsear_write_formula')
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    main()
