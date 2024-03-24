import os.path
import pandas as pd
import nltk
from nltk import word_tokenize
import re
import textstat
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler


def get_dataset_file_paths(dataset_dir, extension):
    file_paths = []

    for root, dirs_list, files_list in os.walk(dataset_dir):
        for file in files_list:
            if os.path.splitext(file)[-1] == extension:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths


def create_output_directories():
    new_folder = "processed_datasets"

    if os.path.exists(new_folder):
        for root, dirs_list, files_list in os.walk(new_folder):
            for file in files_list:
                file_path = os.path.join(root, file)
                os.remove(file_path)

            for directory in dirs_list:
                dir_path = os.path.join(root, directory)
                os.rmdir(dir_path)

            os.rmdir(root)

    else:
        os.mkdir(new_folder)
        os.mkdir("{}\\LIAR_Dataset_processed".format(new_folder))

    return "{}\\LIAR_Dataset_processed".format(new_folder)


def normalise_data(dataframe):
    min_max_scaler = MinMaxScaler()
    dataframe_non_text = pd.DataFrame(dataframe[dataframe.columns[1: 6]])
    dataframe_text = pd.DataFrame(dataframe[dataframe.columns[0]])
    dataframe_label = pd.DataFrame(dataframe[dataframe.columns[6]])

    scaled_data = min_max_scaler.fit_transform(dataframe_non_text)
    scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe_non_text.columns)

    scaled_data_dataframe = dataframe_text.join(scaled_dataframe)
    full_scaled_dataframe = scaled_data_dataframe.join(dataframe_label)

    return full_scaled_dataframe


def get_cleaned_tokens(text):
    stop_words = set(stopwords.words('english'))

    # Replace Non-alpha numeric (symbols, punctuation) with blank spaces.
    text = re.sub("[^a-zA-Z]+", " ", text)

    # Replace single characters with blank spaces.
    text = re.sub(r"(\b\w\b *)", " ", text)

    # Replace multiple blank spaces with one blank space.
    text = re.sub(r"\s+", " ", text)

    # Tokenize text.
    full_text_tokens = [word.lower() for word in word_tokenize(text)]

    # Filter tokens to remove stopwords.
    non_stopword_tokens = [token for token in full_text_tokens if token not in stop_words]

    return full_text_tokens, non_stopword_tokens


def process_LIAR_dataset():
    file_paths = get_dataset_file_paths("LIAR_Dataset", ".tsv")

    LIAR_data = {"text": [], "reading_ease": [], "adjective_count": [], "adverb_count": [],
                 "noun_count": [], "verb_count": [], "label": []}

    for file_path in file_paths:
        file_name = file_path.split("\\")[-1]
        print(f"Processing: {file_name}.")
        file_contents = pd.read_csv(file_path, sep="\t", header=None).dropna()
        file_contents.columns = ["json_id", "label", "statement", "subjects", "speaker", "speaker_job_title",
                                 "state_info", "party_affiliation", "credit_history_count", "barely_true_counts",
                                 "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]

        for index, row in file_contents.iterrows():
            label = row.label
            text = row.statement

            full_text_tokens, non_stopword_tokens = get_cleaned_tokens(text)

            reading_ease = textstat.linsear_write_formula(text)
            tagged = nltk.pos_tag(full_text_tokens, tagset='universal')
            adverb_count = len([word for word in tagged if word[1] == "ADV"])
            noun_count = len([word for word in tagged if word[1] == "NOUN"])
            verb_count = len([word for word in tagged if word[1] == "VERB"])
            adjective_count = len([word for word in tagged if word[1] == "ADJ"])

            label_dict = {"true": 0, "mostly-true": 1, "half-true": 2, "barely-true": 3, "false": 4, "pants-fire": 5}

            LIAR_data["text"].append(" ".join(non_stopword_tokens))
            LIAR_data["reading_ease"].append(reading_ease)
            LIAR_data["adjective_count"].append(adjective_count)
            LIAR_data["verb_count"].append(verb_count)
            LIAR_data["noun_count"].append(noun_count)
            LIAR_data["adverb_count"].append(adverb_count)
            LIAR_data["label"].append(label_dict[label])

    return LIAR_data


def main():
    nltk.download('universal_tagset')
    nltk.download('stopwords')

    LIAR_output_directory = create_output_directories()

    print("Starting to Process LIAR dataset.")
    processed_LIAR_data = process_LIAR_dataset()
    print("Done Processing LIAR dataset.")

    column_names = ["text", "reading_ease", "adjective_count", "adverb_count",
                    "noun_count", "verb_count", "label"]

    LIAR_dataframe = pd.DataFrame(processed_LIAR_data, columns=column_names)
    LIAR_dataframe = normalise_data(LIAR_dataframe)
    LIAR_dataframe.to_csv("{}\\processed_LIAR.csv".format(LIAR_output_directory), index=False)


if __name__ == "__main__":
    main()
