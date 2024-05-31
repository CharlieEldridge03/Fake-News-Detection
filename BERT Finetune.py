import pyarrow as pa
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict, Dataset


def load_liar_data():
    label_dict = {
        0: [True, False, False, False, False, False],
        1: [False, True, False, False, False, False],
        2: [False, False, True, False, False, False],
        3: [False, False, False, True, False, False],
        4: [False, False, False, False, True, False],
        5: [False, False, False, False, False, True]
    }

    LIAR_train_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_train.csv",
                                       index_col=False).dropna()
    LIAR_test_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_test.csv",
                                      index_col=False).dropna()
    LIAR_valid_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR_valid.csv",
                                       index_col=False).dropna()

    LIAR_train_dataframe[["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]] = pd.DataFrame(LIAR_train_dataframe["label"].map(label_dict).tolist(),
                                                          index=LIAR_train_dataframe["label"].index)
    LIAR_test_dataframe[["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]] = pd.DataFrame(LIAR_test_dataframe["label"].map(label_dict).tolist(),
                                                          index=LIAR_test_dataframe["label"].index)
    LIAR_valid_dataframe[["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]] = pd.DataFrame(LIAR_valid_dataframe["label"].map(label_dict).tolist(),
                                                          index=LIAR_valid_dataframe["label"].index)

    LIAR_train_dataframe = LIAR_train_dataframe.drop("label", axis=1)
    LIAR_test_dataframe = LIAR_test_dataframe.drop("label", axis=1)
    LIAR_valid_dataframe = LIAR_valid_dataframe.drop("label", axis=1)

    X_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[0]]
    y_train = LIAR_train_dataframe[LIAR_train_dataframe.columns[22: 28]]

    X_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[0]]
    y_test = LIAR_test_dataframe[LIAR_test_dataframe.columns[22: 28]]

    X_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[0]]
    y_valid = LIAR_valid_dataframe[LIAR_valid_dataframe.columns[22: 28]]

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def load_isot_title_data():
    ISOT_title_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_titles.csv",
        index_col=False).dropna()

    label_dict = {
        0: [True, False],
        1: [False, True]
    }

    ISOT_title_dataframe[["real", "fake"]] = pd.DataFrame(ISOT_title_dataframe["label"].map(label_dict).tolist(),
                                                          index=ISOT_title_dataframe["label"].index)
    ISOT_dataframe = ISOT_title_dataframe.drop("label", axis=1)

    X = ISOT_dataframe[ISOT_dataframe.columns[0]]
    y = ISOT_dataframe[ISOT_dataframe.columns[22: 24]]

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=42)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def load_isot_text_data():
    ISOT_text_dataframe = pd.read_csv(
        "processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT_texts.csv",
        index_col=False).dropna()

    label_dict = {
        0: [True, False],
        1: [False, True]
    }

    ISOT_text_dataframe[["real", "fake"]] = pd.DataFrame(ISOT_text_dataframe["label"].map(label_dict).tolist(),
                                                         index=ISOT_text_dataframe["label"].index)
    ISOT_dataframe = ISOT_text_dataframe.drop("label", axis=1)

    X = ISOT_dataframe[ISOT_dataframe.columns[0]]
    y = ISOT_dataframe[ISOT_dataframe.columns[22: 24]]

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=42)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def create_dataset(X_train, X_test, X_valid, y_train, y_test, y_valid):
    train = X_train.to_frame().join(y_train)
    test = X_test.to_frame().join(y_test)
    valid = X_valid.to_frame().join(y_valid)

    train_dataset = Dataset(pa.Table.from_pandas(train, preserve_index=False))
    test_dataset = Dataset(pa.Table.from_pandas(test, preserve_index=False))
    valid_dataset = Dataset(pa.Table.from_pandas(valid, preserve_index=False))

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": valid_dataset
    })

    return dataset


def preprocess_data(data, tokenizer, labels, max_len):
    # take a batch of texts
    text = data["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_len)
    # add labels
    labels_batch = {k: data[k] for k in data.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}

    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)

    return result


def main():
    X_train, X_test, X_valid, y_train, y_test, y_valid = load_liar_data()

    dataset = create_dataset(X_train, X_test, X_valid, y_train, y_test, y_valid)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    MAX_TOKEN_LEN = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WEIGHT_DECAY = 0.01
    SCORING_METRIC = "accuracy"

    labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    encoded_dataset = dataset.map(preprocess_data,
                                  fn_kwargs={"tokenizer": tokenizer, "labels": labels, "max_len": MAX_TOKEN_LEN},
                                  batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id)
    args = TrainingArguments(
        "bert-liar",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model=SCORING_METRIC,
        push_to_hub=False
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("bert-liar-bestmodel")


if __name__ == "__main__":
    main()
