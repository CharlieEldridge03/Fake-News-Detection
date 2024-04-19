from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, AutoFeatureExtractor, \
    BertModel, AutoModelForCausalLM, AutoConfig, pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
import pyarrow as pa
import pandas as pd
import numpy as np
import accelerate
import torch


def load_liar_data():
    LIAR_dataframe = pd.read_csv("processed_datasets/LIAR_Dataset_processed/processed_LIAR.csv", index_col=False)
    LIAR_dataframe = LIAR_dataframe.dropna()

    X = LIAR_dataframe[LIAR_dataframe.columns[0]]
    y = LIAR_dataframe[LIAR_dataframe.columns[22]]

    return X, y


def load_isot_data():
    ISOT_dataframe = pd.read_csv("processed_datasets/ISOT_Fake_News_Dataset_processed/processed_ISOT.csv", index_col=False)
    ISOT_dataframe = ISOT_dataframe.dropna()

    label_dict = {
        0: [True, False],
        1: [False, True]
    }

    ISOT_dataframe[["real", "fake"]] = pd.DataFrame(ISOT_dataframe["label"].map(label_dict).tolist(), index=ISOT_dataframe["label"].index)
    ISOT_dataframe = ISOT_dataframe.drop("label", axis=1)

    X = ISOT_dataframe[ISOT_dataframe.columns[0]]
    y = ISOT_dataframe[ISOT_dataframe.columns[22: 24]]

    return X, y


def create_dataset(X, y):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    train = X_train.to_frame().join(y_train)
    test = X_test.to_frame().join(y_test)
    val = X_val.to_frame().join(y_val)

    train_dataset = Dataset(pa.Table.from_pandas(train, preserve_index=False))
    test_dataset = Dataset(pa.Table.from_pandas(test, preserve_index=False))
    val_dataset = Dataset(pa.Table.from_pandas(val, preserve_index=False))

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": val_dataset
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
    X, y = load_isot_data()
    dataset = create_dataset(X, y)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #
    # MAX_TOKEN_LEN = 512
    # BATCH_SIZE = 8
    # LEARNING_RATE = 2e-5
    # EPOCHS = 5
    # WEIGHT_DECAY = 0.01
    # SCORING_METRIC = "accuracy"
    #
    # labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
    # id2label = {idx: label for idx, label in enumerate(labels)}
    # label2id = {label: idx for idx, label in enumerate(labels)}
    #
    # encoded_dataset = dataset.map(preprocess_data, fn_kwargs={"tokenizer": tokenizer, "labels": labels, "max_len": MAX_TOKEN_LEN}, batched=True, remove_columns=dataset['train'].column_names)
    # encoded_dataset.set_format("torch")
    #
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
    #                                                            problem_type="multi_label_classification",
    #                                                            num_labels=len(labels),
    #                                                            id2label=id2label,
    #                                                            label2id=label2id)
    # args = TrainingArguments(
    #     "bert-finetuned-fake-news",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=LEARNING_RATE,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     num_train_epochs=EPOCHS,
    #     weight_decay=WEIGHT_DECAY,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=SCORING_METRIC,
    #     push_to_hub=False,
    #     save_total_limit=1,
    # )
    #
    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=encoded_dataset["train"],
    #     eval_dataset=encoded_dataset["validation"],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )

    model = AutoModelForSequenceClassification.from_pretrained("bert-finetuned-isot-articletexts")
    tokenizer = AutoTokenizer.from_pretrained("bert-finetuned-isot-articletexts")
    input_text = "The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own. Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump : - Together, we are MAKING AMERICA GREAT AGAIN! bit.ly/2lnpKaq [1814 EST] - In the East, it could be the COLDEST New Year’s Eve on record. Perhaps we could use a little bit of that good old Global Warming that our Country, but not other countries, was going to pay TRILLIONS OF DOLLARS to protect against. Bundle up! [1901 EST] -- Source link: (bit.ly/2jBh4LU) (bit.ly/2jpEXYR) "
    tokenized_text = tokenizer(input_text,
                               truncation=True,
                               is_split_into_words=False,
                               return_tensors='pt')
    outputs = model(tokenized_text["input_ids"])
    predicted_label = outputs.logits.argmax(-1)
    print(predicted_label)


if __name__ == "__main__":
    main()

