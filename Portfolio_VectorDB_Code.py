"""
Tutorial - https://docs.trychroma.com/getting-started
"""

import chromadb
import pandas as pd
from chromadb import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


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


def create_database(X_train, y_train):
    chroma_client = chromadb.PersistentClient(path="./Vector_DB", settings=Settings(allow_reset=True))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        collection = chroma_client.get_collection("LIAR_DB", embedding_function=embedding_function)
    except ValueError:
        collection = chroma_client.create_collection("LIAR_DB", embedding_function=embedding_function,
                                                     metadata={"hnsw:space": "cosine"}, )

        for i in tqdm(range(len(X_train)), desc="Database Construction: "):
            collection.upsert(
                ids=str(i),
                documents=X_train["text"][i],
                metadatas={"label": str(y_train[i])}
            )

    return collection


def main():
    X_train, X_test, y_train, y_test = load_liar_data()

    collection = create_database(X_train, y_train)
    query_texts = X_test["text"]
    predicted_labels = []

    for i in tqdm(range(len(X_test)), desc="Testing: "):
        result = collection.query(query_texts=query_texts[i], n_results=1)
        predicted_labels.append(int(result['metadatas'][0][0]['label']))

    accuracy = round(accuracy_score(y_test, predicted_labels) * 100, 2)
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    disp.plot()
    plt.title("Accuracy Score: {}%".format(accuracy))
    plt.show()


if __name__ == "__main__":
    main()
