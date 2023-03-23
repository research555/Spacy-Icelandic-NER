import os
import pdb
import pprint
import re
import json
import pickle
import string
import pdb
import random
from tqdm import tqdm

import spacy_transformers

"""
org_data:

sentence = data[key]['sentence']
words: for word in item:
        word['word']
[   
    {
        sentence: str,
        words: [
                {
                word: str,
                ent: str
                 },
                {
                word: str,
                ent: str
                },
                .
                .
                .
    .
    .
    .



]



"""


class Preprocess:
    """
    A class for preprocessing text data.

    Attributes:
    data_dir (str): The directory where the data is stored.
    training_data (dict): A dictionary to hold the training data.
    index (int): The index of the current data item.
    combined (dict): A dictionary to hold the combined data.
    sentences (list): A list to hold the processed sentences.
    punctuation (str): A string containing punctuation characters to be removed from the text.
    stopwords (list): A list of stop words to be removed from the text.
    formatted_data (dict): A dictionary to hold the formatted data.
    combined_and_cleaned (list): A list to hold the combined and cleaned data.
    """

    def __init__(self):

        self.data_dir = u"/MIM-GOLD-2_0"
        self.training_data = {}
        self.index = 0
        self.combined = {}
        self.sentences = []
        self.punctuation = '"#&\'()*+,:<=>@[\\]^_`{|}~»«'
        self.stopwords = self.stopwords()
        self.formatted_data = {}
        self.combined_and_cleaned = []

    def clean_txt(self, file, combined: bool = False) -> list:
        """
        Cleans the input text file by removing unnecessary characters and formatting the data.

        Args:
        file (file object): A file object containing the text data to be cleaned.
        combined (bool, optional): A flag to indicate whether to return combined and cleaned data.

        Returns:
        cleaned_data (list): A list of dictionaries containing the cleaned data.
        """
        text = file.read()
        text = text.strip()
        text = text.split('\n')  # split word for word
        text = [item for item in text if item != ' ' and item != '' and item != '\n']  # remove empty strings
        text = [item.split('\t') for item in text]  # split into word and entity
        for i in tqdm(range(len(text))):
            word = text[i][0]
            ent = text[i][1]

            word = re.sub('<.*?>', '', word) #remove tags ALL
            word = word.translate(str.maketrans('', '', self.punctuation)) # ALL
            word = word.strip() # ALL
            word = word.lower() #ALL
            word = re.sub(r'https?://\S+', '', word)  # remove urls # ALL
            word = re.sub(r'www\.\S+', '', word)  # remove urls # ALL
            word = re.sub(r'@\S+', '', word)  # remove @mentions and emails #ALL
            text[i][0] = word

        #pdb.set_trace()
        #text = [item for item in text if word != ''
        #        and word not in self.stopwords
        #        and word not in self.punctuation
        #        and word != ' '
        #        and len(item) == 2]

        text = [item for item in text if item[0] != '']  # remove empty strings
        text = [item for item in text if item[0] != ' ']  # remove empty strings
        text = [item for item in text if item[0] not in self.punctuation]  # remove punctuation
        text = [item for item in text if len(item) == 2]  # normalize length of list
        if combined:
            self.combined_and_cleaned.extend({'word': item[0], 'ent': item[1]} for item in text)
            return self.combined_and_cleaned

        return [{'word': word, 'ent': ent} for item in text]

    @staticmethod
    def stopwords() -> list:
        """
        Reads stop words from a file and returns them as a list.

        Returns:
        stopwords (list): A list of stop words to be removed from the text.
        """

        with open(r'stop_words.txt', 'r', encoding='utf-8') as file:
            return file.read().splitlines()

    @staticmethod
    def strict_encode_utf8(self, data: list, word: str = None) -> list:
        """
        Encodes the input data as strict UTF-8.

        Args:
        data (list): The data to be encoded.
        word (str, optional): A string to be encoded.

        Returns:
        encoded_data (list): The encoded data.
        """

        if word:
            return word.encode('utf-8')
        for item in data:
            item['sentence'] = item['sentence'].encode('utf-8')
            for word in item['words']:
                word['word'] = word['word'].encode('utf-8')
        return data

    def oversample_underrepresented_labels(self, ratios) -> dict:
        """
        Oversamples underrepresented entity labels in a dataset to balance the dataset.

        Args:
            data (dict): A dictionary where each key is a unique identifier for a piece of text
                and each value is a dictionary with two keys: "text" and "entities". "text" is a string
                containing the raw text and "entities" is a list of tuples where each tuple contains
                the start and end positions of an entity mention in the text, along with the label of
                the entity.
            ratios (dict): A dictionary where each key is an entity label that needs oversampling and
                each value is the desired oversampling ratio. The ratio is the desired number of
                sentences with the given label after oversampling divided by the number of sentences
                with the given label before oversampling.

        Returns:
            A new dictionary with the same structure as the input data, but with additional sentences
            generated to balance the dataset.
        """

        for label, ratio in ratios.items():
            if ratio <= 1:
                continue

            label_sentences = []
            for key, value in self.formatted_data.items():
                entities = value["entities"]
                for start, end, ent_label in entities:
                    if ent_label == label:
                        label_sentences.append(value)
                        break

            if not label_sentences:
                continue

            num_label_sentences = len(label_sentences)
            num_new_sentences = int(ratio * num_label_sentences) - num_label_sentences

            for i in range(num_new_sentences):
                sentence = random.choice(label_sentences)
                new_sentence = sentence.copy()
                new_sentence_entities = new_sentence["entities"]
                label_entities = [(start, end, ent_label) for start, end, ent_label in new_sentence_entities if
                                  ent_label == label]
                new_entities = new_sentence_entities.copy()
                if label_entities:
                    random.shuffle(label_entities)
                    new_entities += label_entities[:2 - len(new_entities)]
                    new_sentence["entities"] = new_entities
                    max_key = max(self.formatted_data.keys())
                    new_key = max_key + 1
                    self.formatted_data[new_key] = new_sentence


        return self.formatted_data

    def format_data(self, organized_data: list, reduce_o: bool = False) -> dict:
        """
        Format the organized data into a dictionary of text with their corresponding entities.

        Args:
            organized_data (list): List of organized data containing sentences and their corresponding words.
            reduce_o (bool, optional): Flag indicating whether to reduce the 'O' entities in the formatted data.
                                        Defaults to False.

        Returns:
            dict: A dictionary of text with their corresponding entities.
        """
        #pdb.set_trace()
        for i, item in tqdm(enumerate(organized_data)):
            self.index += 1
            entities = []
            for word in item['words']:
                escaped_word = re.escape(word['word'])
                match = ((m.start(0), m.end(0), word['ent']) for m in re.finditer(escaped_word, item['sentence']))
                entities.extend(match)
            self.formatted_data[self.index] = {'text': item['sentence'], 'entities': list(entities)}
        if reduce_o:
            self.formatted_data = {index: entities_dict for index, entities_dict in self.formatted_data.items()
                        if not all(ent[2] == 'O' for ent in entities_dict['entities'])}
        return self.formatted_data

    def organize_data(self, combined=False) -> list:
        """
            Organize the training data into a list of sentences.

            Args:
                combined (bool, optional): Flag indicating whether the training data is already combined. Defaults to False.

            Returns:
                list: A list of sentences.
            """

        sentences = []
        if combined:
            sentences = self.sentences
        current_sentence = {'sentence': '', 'words': []}

        for i, words in tqdm(enumerate(self.training_data.values())):
            for sub_list in words:
                if sub_list['word'] == '.' \
                        or sub_list['word'] == '?' \
                        or sub_list['word'] == '!' \
                        or sub_list['word'] == ';':
                    sentences.append(current_sentence)
                    current_sentence = {'sentence': '', 'words': []}
                else:
                    current_sentence['words'].append(sub_list)
                    current_sentence['sentence'] = ' '.join([word['word'] for word in current_sentence['words']])

        return sentences

    def count_ents(self, data: dict = None, formatted: bool = False) -> dict:
        """
        Count the number of entities in the data.

        Args:
            data (dict, optional): A dictionary of data to count the entities from. Defaults to None.
            formatted (bool, optional): Flag indicating whether to count entities from the formatted data. Defaults to False.

        Returns:
            dict: A dictionary containing the count of each entity type.
        """

        self.number_ents = {
            'O': 0,
            'B-Person': 0,
            'I-Person': 0,
            'B-Location': 0,
            'I-Location': 0,
            'B-Organization': 0,
            'I-Organization': 0,
            'B-Miscellaneous': 0,
            'I-Miscellaneous': 0,
            'B-Date': 0,
            'I-Date': 0,
            'B-Time': 0,
            'I-Time': 0,
            'B-Percent': 0,
            'I-Percent': 0,
            'B-Money': 0,
            'I-Money': 0
        }

        if formatted:
            data = self.formatted_data
        for index, item in tqdm(enumerate(data.values())):
            for ent in item['entities']:
                self.number_ents[ent[2]] += 1
        return self.number_ents

    def alter_ents(self, reduce_o: bool = False, data: dict = None) -> dict:
        """
            Alter the entities in the formatted data by removing the 'O' entities if the 'reduce_o' flag is set to True.

            Args:
                reduce_o (bool, optional): Flag indicating whether to remove the 'O' entities. Defaults to False.
                data (dict, optional): A dictionary of data to alter the entities from. Defaults to None.

            Returns:
                dict: A dictionary of data with the altered entities.
            """
        if data:
            self.formatted_data = data
        rows_to_delete = []

        for index, item in self.formatted_data.items():
            if reduce_o:
                entities = [ent for start, end, ent in item['entities']]
                if all(ent == 'O' for ent in entities) or not entities:
                    rows_to_delete.append(index)
        for key in rows_to_delete:
            del self.formatted_data[key]
        return self.formatted_data

    def combine_all_to_pkl(self) -> str:
        """
        Combine all the PKL files into a single file called "combined.pkl".

        Returns:
            str: A string indicating that the files have been combined.
        """

        with open('../train_data/combined.pkl', 'wb') as file:
            for filename in os.listdir('pkl/pkl_with_O'):
                with open(os.path.join('pkl/pkl_with_O', filename), 'rb') as pkl_file:
                    self.combined[filename] = pickle.load(pkl_file)
            pickle.dump(self.combined, file)
            return 'files combined'

    def generate_training_data(self, cleaned_txt) -> dict:
        """
        Generate the training data from the cleaned text.

        Args:
            cleaned_txt (str): The cleaned text to generate the training data from.

        Returns:
            dict: A dictionary containing the training data.
        """
        for filename in tqdm(os.listdir(self.data_dir)):
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as file:
                self.training_data[filename] = cleaned_txt
        return self.training_data

    def save_pkl(self, dir: str = 'pkl/pkl_with_O') -> str:
        """
        Save the formatted data as PKL files.

        Args:
            dir (str, optional): The directory where the PKL files should be saved. Defaults to 'pkl/pkl_with_O'.

        Returns:
            str: A string indicating that the PKL files have been saved.
        """
        for filename, data in self.training_data.items():
            with open(f'{dir}/{filename[0:-4]}.pkl', 'wb') as file:
                formatted_data = self.format_data(self.organize_data(data))
                pickle.dump(formatted_data, file)
        return 'pkl files saved'

    def reduce_o_tags(self, o_per_sentence: int = 1, data: dict = False) -> dict:
        """
        Reduce the 'O' entities in the data by randomly selecting 'o_per_sentence' 'O' entities per sentence.

        Args:
            o_per_sentence (int, optional): The number of 'O' entities to keep per sentence. Defaults to 1.
            data (dict, optional): A dictionary of data to reduce the 'O' entities from. Defaults to False.

        Returns:
            dict: A dictionary of data with the reduced 'O' entities.
        """
        if data:
            self.formatted_data = data
        for index, item in tqdm(data.items()):
            entities = item['entities']
            o_tags = [ent for ent in entities if ent[2] == 'O']
            random.shuffle(o_tags)
            if len(o_tags) > o_per_sentence:
                for ent in o_tags[:len(o_tags) - o_per_sentence]:
                    if ent in self.formatted_data[index]['entities']:
                        entities.remove(ent)
        return self.formatted_data

    def reduce_features(self, number: int, data: dict = None) -> dict:
        """
        Reduces the number of features in the input data dictionary to the given percentage.

        Args:
            number (int): The percentage of features to keep (integer between 0 and 100).
            data (dict, optional): The input data dictionary to reduce. If not provided, uses the
                formatted_data attribute of the class instance.

        Returns:
            dict: The reduced data dictionary.
        """

        if data:
            self.formatted_data = data
        keys = list(self.formatted_data.keys())
        print(len(keys))
        random.shuffle(keys)
        keys_to_keep = len(keys) - round(len(keys) * (number / 100))
        keys = keys[0:keys_to_keep]

        self.formatted_data = {key: self.formatted_data[key] for key in keys}
        print(len(self.formatted_data))

        return self.formatted_data


    def split_training_test_set(self, train: int, test: int, data: dict = None) -> tuple:
        if data:
            self.formatted_data = data
        if train + test != 100:
            raise ValueError('train and test must add up to 100')

        length = len(self.formatted_data)
        keys = list(self.formatted_data.keys())
        random.shuffle(keys)
        train_keys = keys[:int(length * train / 100)]
        test_keys = keys[int(length * train / 100):]
        train_set = {key: self.formatted_data[key] for key in train_keys}
        test_set = {key: self.formatted_data[key] for key in test_keys}

        return train_set, test_set

    def to_splits(self) -> None:
        """
        Generate and save different training-test splits of the data by reducing the number of features,
        altering entities, and reducing 'O' tags. The splits are saved as pickle files in the specified directory structure.

        Returns:
        None
        """

        train = 0
        test = 0
        with open(r'train_data\try6\combined_all_files_fr_pre_try6_low_o.pkl', 'rb') as file:
            f_data = pickle.load(file)

        for i in range(10, 100, 10):
            train = 100 - i
            test = i

            for i in tqdm(range(5, 95, 5)):
                parent_directory = r'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits'
                sub1 = f'train{train}test{test}'
                sub2 = f'deleted{i}'
                directory = os.path.join(parent_directory, sub1, sub2)

                if not os.path.exists(os.path.join(parent_directory, sub1)):
                    os.mkdir(os.path.join(parent_directory, sub1))

                if not os.path.exists(directory):
                    os.mkdir(directory)

                red = self.reduce_features(i, f_data)
                print(f'features reduced to {i} percent')
                alt = self.alter_ents(reduce_o=True, data=red)
                print('empty O tags removed')
                low_o = self.reduce_o_tags(1, data=alt)
                print('O tags reduced to 1 per sentence')

                with open(rf'{directory}/combined_dataset_{i}_percent_removed.pkl', 'wb') as file:
                    pickle.dump(low_o, file)
                    print(f'combined_dataset_{i}_percent_removed.pkl saved, length {len(low_o)}')

                train_set, test_set = self.split_training_test_set(train, test)
                with open(rf'{directory}/TRAIN_combined_{i}_percent_{str(train)}_{str(test)}.pkl', 'wb') as file:
                    pickle.dump(train_set, file)
                    print(f'train_set saved, length {len(train_set)}')

                with open(rf'{directory}/TEST_combined_{i}_percent_{str(train)}_{str(test)}.pkl', 'wb') as file:
                    pickle.dump(test_set, file)
                    print(f'test_set saved, length {len(test_set)}')



"""ent_ratio = {
            'O': 0,
            'B-Person': 0,
            'I-Person': 2,
            'B-Location':10,
            'I-Location': 10,
            'B-Organization': 2,
            'I-Organization': 3,
            'B-Miscellaneous': 2,
            'I-Miscellaneous': 2,
            'B-Date': 0,
            'I-Date': 0,
            'B-Time': 10,
            'I-Time': 10,
            'B-Percent': 0,
            'I-Percent': 0,
            'B-Money': 20,
            'I-Money': 20
        }"""

if __name__ == '__main__':

    with open(r'/train_data/splits/train60test40/deleted90/TRAIN_combined_90_percent_60_40.pkl', 'rb') as file:
        train_set = pickle.load(file)


    total_items = len(train_set)
    print(total_items)
    dev_ratio = 15/100
    train_keys = list(train_set.keys())
    train_keys = train_keys[0:(int(len(train_keys)) * (1 - dev_ratio))]
    dev_keys = train_keys[(int(len(train_keys)) * dev_ratio):]
    print(len(train_keys))
    print(len(dev_keys))