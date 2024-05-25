"""

"""


from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import textstat
import os.path
import nltk
import re


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
        os.mkdir("{}\\ISOT_Fake_News_Dataset_processed".format(new_folder))
        os.mkdir("{}\\LIAR_Dataset_processed".format(new_folder))

    return "{}\\ISOT_Fake_News_Dataset_processed".format(new_folder), "{}\\LIAR_Dataset_processed".format(new_folder)


def normalise_data(dataframe):
    min_max_scaler = MinMaxScaler()
    dataframe_non_text = pd.DataFrame(dataframe[dataframe.columns[1: 22]])
    dataframe_text = pd.DataFrame(dataframe[dataframe.columns[0]])
    dataframe_label = pd.DataFrame(dataframe[dataframe.columns[22]])

    scaled_data = min_max_scaler.fit_transform(dataframe_non_text)
    scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe_non_text.columns)

    scaled_data_dataframe = dataframe_text.join(scaled_dataframe)
    full_scaled_dataframe = scaled_data_dataframe.join(dataframe_label)

    return full_scaled_dataframe


def get_cleaned_tokens(text):
    stop_words = set(stopwords.words('english'))

    # Replace Non-alpha numeric (symbols, punctuation) with blank spaces.
    text = re.sub("[^a-zA-Z]+", " ", text)

    # Remove single characters with blank spaces.
    text = re.sub(r"(\b\w\b *)", " ", text)

    # Replace multiple blank spaces with one blank space.
    text = re.sub(r"\s+", " ", text)

    # Tokenize text.
    full_text_tokens = [word.lower() for word in word_tokenize(text)]

    # Filter tokens to remove stopwords.
    non_stopword_tokens = [token for token in full_text_tokens if token not in stop_words]

    return full_text_tokens, non_stopword_tokens


def process_ISOT_dataset():
    file_paths = get_dataset_file_paths("datasets\\ISOT_Fake_News_Dataset", ".csv")
    stop_words = set(stopwords.words('english'))

    ISOT_title_data = {"text": [], "conjunction_count": [], "pronoun_count": [], "adjective_count": [], "adverb_count": [],
                 "noun_count": [], "verb_count": [], "flesch_reading_ease": [], "flesch_kincaid_grade_level": [],
                 "gunning_fog_scale": [], "coleman_liau_index": [], "linsear_write_formula": [],
                 "dale_chall_readability": [], "mcalpine_eflaw_readability": [], "reading_time": [],
                 "syllable_count": [], "lexicon_count": [], "polysyllable_count": [], "monosyllable_count": [],
                 "stopword_count": [], "present_tense_count": [], "past_tense_count": [], "label": []}

    ISOT_text_data = {"text": [], "conjunction_count": [], "pronoun_count": [], "adjective_count": [], "adverb_count": [],
                 "noun_count": [], "verb_count": [], "flesch_reading_ease": [], "flesch_kincaid_grade_level": [],
                 "gunning_fog_scale": [], "coleman_liau_index": [], "linsear_write_formula": [],
                 "dale_chall_readability": [], "mcalpine_eflaw_readability": [], "reading_time": [],
                 "syllable_count": [], "lexicon_count": [], "polysyllable_count": [], "monosyllable_count": [],
                 "stopword_count": [], "present_tense_count": [], "past_tense_count": [], "label": []}

    for file_path in file_paths:
        file_name = file_path.split("\\")[-1]
        print(f"Processing: {file_name}.")

        file_contents = pd.read_csv(file_path, sep=",")

        for _, row in file_contents.iterrows():
            title = row.title
            text = row.text

            full_title_tokens, non_stopword_tokens_title = get_cleaned_tokens(title)
            full_text_tokens, non_stopword_tokens_text = get_cleaned_tokens(text)

            non_stopword_tokens_title = " ".join(non_stopword_tokens_title)
            non_stopword_tokens_text = " ".join(non_stopword_tokens_text)

            flesch_reading_ease_title = textstat.flesch_reading_ease(title)
            flesch_kincaid_grade_level_title = textstat.flesch_kincaid_grade(title)
            gunning_fog_scale_title = textstat.gunning_fog(title)
            coleman_liau_index_title = textstat.coleman_liau_index(title)
            linsear_write_formula_title = textstat.linsear_write_formula(title)
            dale_chall_readability_title = textstat.dale_chall_readability_score(title)
            mcalpine_eflaw_readability_title = textstat.mcalpine_eflaw(title)
            reading_time_title = textstat.reading_time(title, ms_per_char=14.69)
            syllable_count_title = textstat.syllable_count(title)
            lexicon_count_title = textstat.lexicon_count(title, removepunct=True)
            polysyllable_count_title = textstat.polysyllabcount(title)
            monosyllable_count_title = textstat.monosyllabcount(title)

            flesch_reading_ease_text = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade_level_text = textstat.flesch_kincaid_grade(text)
            gunning_fog_scale_text = textstat.gunning_fog(text)
            coleman_liau_index_text = textstat.coleman_liau_index(text)
            linsear_write_formula_text = textstat.linsear_write_formula(text)
            dale_chall_readability_text = textstat.dale_chall_readability_score(text)
            mcalpine_eflaw_readability_text = textstat.mcalpine_eflaw(text)
            reading_time_text = textstat.reading_time(text, ms_per_char=14.69)
            syllable_count_text = textstat.syllable_count(text)
            lexicon_count_text = textstat.lexicon_count(text, removepunct=True)
            polysyllable_count_text = textstat.polysyllabcount(text)
            monosyllable_count_text = textstat.monosyllabcount(text)

            tagged_title = nltk.pos_tag(full_title_tokens, tagset='universal')
            adverb_count_title = len([word for word in tagged_title if word[1] == "ADV"])
            noun_count_title = len([word for word in tagged_title if word[1] == "NOUN"])
            verb_count_title = len([word for word in tagged_title if word[1] == "VERB"])
            adjective_count_title = len([word for word in tagged_title if word[1] == "ADJ"])
            pronoun_count_title = len([word for word in tagged_title if word[1] == "PRON"])
            conjunction_count_title = len([word for word in tagged_title if word[1] == "CONJ"])
            stopword_count_title = len([word for word in full_title_tokens if word in stop_words])
            tagged_title = nltk.pos_tag(full_title_tokens)
            present_tense_count_title = len([word for word in tagged_title if word[1] == "VBG" or word[1] == "VBP" or word[1] == "VBZ"])
            past_tense_count_title = len([word for word in tagged_title if word[1] == "VBD" or word[1] == "VBN"])

            tagged_text = nltk.pos_tag(full_text_tokens, tagset='universal')
            adverb_count_text = len([word for word in tagged_text if word[1] == "ADV"])
            noun_count_text = len([word for word in tagged_text if word[1] == "NOUN"])
            verb_count_text = len([word for word in tagged_text if word[1] == "VERB"])
            adjective_count_text = len([word for word in tagged_text if word[1] == "ADJ"])
            pronoun_count_text = len([word for word in tagged_text if word[1] == "PRON"])
            conjunction_count_text = len([word for word in tagged_text if word[1] == "CONJ"])
            stopword_count_text = len([word for word in full_text_tokens if word in stop_words])
            tagged_text = nltk.pos_tag(full_text_tokens)
            present_tense_count_text = len([word for word in tagged_text if word[1] == "VBG" or word[1] == "VBP" or word[1] == "VBZ"])
            past_tense_count_text = len([word for word in tagged_text if word[1] == "VBD" or word[1] == "VBN"])

            if file_name == "Fake.csv":
                label = 1
            else:
                label = 0

            ISOT_title_data["text"].append(non_stopword_tokens_title)
            ISOT_title_data["conjunction_count"].append(conjunction_count_title)
            ISOT_title_data["pronoun_count"].append(pronoun_count_title)
            ISOT_title_data["adjective_count"].append(adjective_count_title)
            ISOT_title_data["verb_count"].append(verb_count_title)
            ISOT_title_data["noun_count"].append(noun_count_title)
            ISOT_title_data["adverb_count"].append(adverb_count_title)
            ISOT_title_data["flesch_reading_ease"].append(flesch_reading_ease_title)
            ISOT_title_data["flesch_kincaid_grade_level"].append(flesch_kincaid_grade_level_title)
            ISOT_title_data["gunning_fog_scale"].append(gunning_fog_scale_title)
            ISOT_title_data["coleman_liau_index"].append(coleman_liau_index_title)
            ISOT_title_data["linsear_write_formula"].append(linsear_write_formula_title)
            ISOT_title_data["dale_chall_readability"].append(dale_chall_readability_title)
            ISOT_title_data["mcalpine_eflaw_readability"].append(mcalpine_eflaw_readability_title)
            ISOT_title_data["reading_time"].append(reading_time_title)
            ISOT_title_data["syllable_count"].append(syllable_count_title)
            ISOT_title_data["lexicon_count"].append(lexicon_count_title)
            ISOT_title_data["polysyllable_count"].append(polysyllable_count_title)
            ISOT_title_data["monosyllable_count"].append(monosyllable_count_title)
            ISOT_title_data["stopword_count"].append(stopword_count_title)
            ISOT_title_data["present_tense_count"].append(present_tense_count_title)
            ISOT_title_data["past_tense_count"].append(past_tense_count_title)
            ISOT_title_data["label"].append(label)

            ISOT_text_data["text"].append(non_stopword_tokens_text)
            ISOT_text_data["conjunction_count"].append(conjunction_count_text)
            ISOT_text_data["pronoun_count"].append(pronoun_count_text)
            ISOT_text_data["adjective_count"].append(adjective_count_text)
            ISOT_text_data["verb_count"].append(verb_count_text)
            ISOT_text_data["noun_count"].append(noun_count_text)
            ISOT_text_data["adverb_count"].append(adverb_count_text)
            ISOT_text_data["flesch_reading_ease"].append(flesch_reading_ease_text)
            ISOT_text_data["flesch_kincaid_grade_level"].append(flesch_kincaid_grade_level_text)
            ISOT_text_data["gunning_fog_scale"].append(gunning_fog_scale_text)
            ISOT_text_data["coleman_liau_index"].append(coleman_liau_index_text)
            ISOT_text_data["linsear_write_formula"].append(linsear_write_formula_text)
            ISOT_text_data["dale_chall_readability"].append(dale_chall_readability_text)
            ISOT_text_data["mcalpine_eflaw_readability"].append(mcalpine_eflaw_readability_text)
            ISOT_text_data["reading_time"].append(reading_time_text)
            ISOT_text_data["syllable_count"].append(syllable_count_text)
            ISOT_text_data["lexicon_count"].append(lexicon_count_text)
            ISOT_text_data["polysyllable_count"].append(polysyllable_count_text)
            ISOT_text_data["monosyllable_count"].append(monosyllable_count_text)
            ISOT_text_data["stopword_count"].append(stopword_count_text)
            ISOT_text_data["present_tense_count"].append(present_tense_count_text)
            ISOT_text_data["past_tense_count"].append(past_tense_count_text)
            ISOT_text_data["label"].append(label)

    return ISOT_title_data, ISOT_text_data


def process_LIAR_dataset():
    file_paths = get_dataset_file_paths("datasets/LIAR_Dataset", ".tsv")
    stop_words = set(stopwords.words('english'))

    LIAR_datasets = []

    for file_path in file_paths:
        LIAR_data = {"text": [], "conjunction_count": [], "pronoun_count": [], "adjective_count": [], "adverb_count": [],
                     "noun_count": [], "verb_count": [], "flesch_reading_ease": [], "flesch_kincaid_grade_level": [],
                     "gunning_fog_scale": [], "coleman_liau_index": [], "linsear_write_formula": [],
                     "dale_chall_readability": [], "mcalpine_eflaw_readability": [], "reading_time": [],
                     "syllable_count": [], "lexicon_count": [], "polysyllable_count": [], "monosyllable_count": [],
                     "stopword_count": [], "present_tense_count": [], "past_tense_count": [], "label": []}

        file_name = file_path.split("\\")[-1]
        print(f"Processing: {file_name}.")
        file_contents = pd.read_csv(file_path, sep="\t", header=None)
        file_contents.columns = ["json_id", "label", "statement", "subjects", "speaker", "speaker_job_title",
                                 "state_info", "party_affiliation", "credit_history_count", "barely_true_counts",
                                 "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]

        for _, row in file_contents.iterrows():
            label = row.label
            text = row.statement

            full_text_tokens, non_stopword_tokens = get_cleaned_tokens(text)
            non_stopword_tokens = " ".join(non_stopword_tokens)

            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text)
            gunning_fog_scale = textstat.gunning_fog(text)
            coleman_liau_index = textstat.coleman_liau_index(text)
            linsear_write_formula = textstat.linsear_write_formula(text)
            dale_chall_readability = textstat.dale_chall_readability_score(text)
            mcalpine_eflaw_readability = textstat.mcalpine_eflaw(text)
            reading_time = textstat.reading_time(text, ms_per_char=14.69)
            syllable_count = textstat.syllable_count(text)
            lexicon_count = textstat.lexicon_count(text, removepunct=True)
            polysyllable_count = textstat.polysyllabcount(text)
            monosyllable_count = textstat.monosyllabcount(text)

            tagged = nltk.pos_tag(full_text_tokens, tagset='universal')
            adverb_count = len([word for word in tagged if word[1] == "ADV"])
            noun_count = len([word for word in tagged if word[1] == "NOUN"])
            verb_count = len([word for word in tagged if word[1] == "VERB"])
            adjective_count = len([word for word in tagged if word[1] == "ADJ"])
            pronoun_count = len([word for word in tagged if word[1] == "PRON"])
            conjunction_count = len([word for word in tagged if word[1] == "CONJ"])

            stopword_count = len([word for word in full_text_tokens if word in stop_words])
            tagged = nltk.pos_tag(full_text_tokens)
            present_tense_count = len( [word for word in tagged if word[1] == "VBG" or word[1] == "VBP" or word[1] == "VBZ"])
            past_tense_count = len([word for word in tagged if word[1] == "VBD" or word[1] == "VBN"])

            label_dict = {"true": 0, "mostly-true": 1, "half-true": 2, "barely-true": 3, "false": 4, "pants-fire": 5}

            LIAR_data["text"].append(non_stopword_tokens)
            LIAR_data["conjunction_count"].append(conjunction_count)
            LIAR_data["pronoun_count"].append(pronoun_count)
            LIAR_data["adjective_count"].append(adjective_count)
            LIAR_data["verb_count"].append(verb_count)
            LIAR_data["noun_count"].append(noun_count)
            LIAR_data["adverb_count"].append(adverb_count)
            LIAR_data["flesch_reading_ease"].append(flesch_reading_ease)
            LIAR_data["flesch_kincaid_grade_level"].append(flesch_kincaid_grade_level)
            LIAR_data["gunning_fog_scale"].append(gunning_fog_scale)
            LIAR_data["coleman_liau_index"].append(coleman_liau_index)
            LIAR_data["linsear_write_formula"].append(linsear_write_formula)
            LIAR_data["dale_chall_readability"].append(dale_chall_readability)
            LIAR_data["mcalpine_eflaw_readability"].append(mcalpine_eflaw_readability)
            LIAR_data["reading_time"].append(reading_time)
            LIAR_data["syllable_count"].append(syllable_count)
            LIAR_data["lexicon_count"].append(lexicon_count)
            LIAR_data["polysyllable_count"].append(polysyllable_count)
            LIAR_data["monosyllable_count"].append(monosyllable_count)
            LIAR_data["stopword_count"].append(stopword_count)
            LIAR_data["present_tense_count"].append(present_tense_count)
            LIAR_data["past_tense_count"].append(past_tense_count)
            LIAR_data["label"].append(label_dict[label])

        LIAR_datasets.append(LIAR_data)

    return LIAR_datasets


def main():
    nltk.download('universal_tagset')
    nltk.download('stopwords')

    ISOT_output_directory, LIAR_output_directory = create_output_directories()
    print("Starting to Process ISOT dataset.")
    processed_ISOT_title_data, processed_ISOT_text_data = process_ISOT_dataset()
    print("Done Processing ISOT dataset.\n")

    print("Starting to Process LIAR dataset.")
    processed_LIAR_test_data, processed_LIAR_train_data, processed_LIAR_valid_data = process_LIAR_dataset()
    print("Done Processing LIAR dataset.")

    column_names = ["text", "adjective_count", "adverb_count",
                    "noun_count", "verb_count", "flesch_reading_ease", "flesch_kincaid_grade_level",
                    "gunning_fog_scale", "coleman_liau_index",
                    "linsear_write_formula", "dale_chall_readability",
                    "mcalpine_eflaw_readability", "reading_time", "syllable_count",
                    "lexicon_count", "polysyllable_count", "monosyllable_count", "conjunction_count",
                    "pronoun_count", "stopword_count", "present_tense_count", "past_tense_count", "label"]

    ISOT_title_dataframe = pd.DataFrame(processed_ISOT_title_data, columns=column_names)
    ISOT_text_dataframe = pd.DataFrame(processed_ISOT_text_data, columns=column_names)
    LIAR_test_dataframe = pd.DataFrame(processed_LIAR_test_data, columns=column_names)
    LIAR_train_dataframe = pd.DataFrame(processed_LIAR_train_data, columns=column_names)
    LIAR_valid_dataframe = pd.DataFrame(processed_LIAR_valid_data, columns=column_names)

    ISOT_title_dataframe = normalise_data(ISOT_title_dataframe)
    ISOT_text_dataframe = normalise_data(ISOT_text_dataframe)
    LIAR_test_dataframe = normalise_data(LIAR_test_dataframe)
    LIAR_train_dataframe = normalise_data(LIAR_train_dataframe)
    LIAR_valid_dataframe = normalise_data(LIAR_valid_dataframe)

    ISOT_title_dataframe.to_csv("{}\\processed_ISOT_titles.csv".format(ISOT_output_directory), index=False)
    ISOT_text_dataframe.to_csv("{}\\processed_ISOT_texts.csv".format(ISOT_output_directory), index=False)
    LIAR_test_dataframe.to_csv("{}\\processed_LIAR_test.csv".format(LIAR_output_directory), index=False)
    LIAR_train_dataframe.to_csv("{}\\processed_LIAR_train.csv".format(LIAR_output_directory), index=False)
    LIAR_valid_dataframe.to_csv("{}\\processed_LIAR_valid.csv".format(LIAR_output_directory), index=False)


if __name__ == "__main__":
    main()
