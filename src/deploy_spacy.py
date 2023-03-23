import pdb

import pkl
from spacy import blank
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import pickle
from pprint import PrettyPrinter
from spacy.util import filter_spans
import os

class Analysis:

    def generate_sample_data(self, data, ratios):
        sample_data = {}
        for tag, ratio in ratios.items():
            sample_size = int(len(data) * ratio)
            sample_data[tag] = []
            for key, value in data.items():
                if value['entities'].count((0, 0, 'O')) > 0:
                    continue
                if len(sample_data[tag]) >= sample_size:
                    break
                if any([True if entity[2] == tag else False for entity in value['entities']]):
                    sample_data[tag].append(key)
        return sample_data

    def mean_o_tags_per_sentence(self, data):
        o_counts = []
        total_words = 0
        total_sentences = 0
        for _, value in data.items():
            entities = value['entities']
            o_count = len([entity for entity in entities if entity[2] == 'O'])
            o_counts.append(o_count)
            words = len(value['text'].split())
            total_words += words
            total_sentences += 1
        mean_words_per_sentence = total_words / total_sentences
        mean_o_tags = sum(o_counts) / len(o_counts)
        return mean_o_tags, mean_words_per_sentence


class DeployModel:
    """
    Class containing methods to generate Spacy output files using the training data.
    """

    def __init__(self):

        self.nlp = blank('is')
        self.nlp.add_pipe('ner')
        self.parent_directory = r'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits'
        self.config_path = r'/config.cfg'
        self.doc_bin_path = None
        self.model_output_path = None


    def get_train_data(self) -> dict:
        """
        Method to get the training data and generate Spacy output files.

        Returns:
            None
        """

        for splits_directory in os.listdir(self.parent_directory):
            splits_directory = os.path.join(self.parent_directory, splits_directory)
            for sub_dir in tqdm(os.listdir(splits_directory)):
                train_file_directory = os.path.join(splits_directory, sub_dir)
                for file in os.listdir(train_file_directory):
                    if file.startswith('TRAIN'):
                        file_name = file
                        full_train_path = os.path.join(train_file_directory, file)
                        with open(full_train_path, 'rb') as f:
                            train_data = pickle.load(f)
                            print(f'Loaded {file_name} from {train_file_directory}')
                            print(len(train_data))
                            doc_bin = self.generate_doc_bin(train_data, train_file_directory, file_name)
                            output = self.generate_spacy_output(f'{os.path.join(train_file_directory, file_name)[:-4]}.spacy', train_file_directory, self.config_path)
                            print(output)
                            f.close()

    def generate_doc_bin(self, data, path: str, file_name: str) -> DocBin:
        """
        Generates a Spacy doc_bin object from training data and returns it.
        """

        doc_bin = DocBin()

        for training_item in tqdm(data.values()):
            text = training_item['text']
            labels = training_item['entities']
            doc = self.nlp.make_doc(text)
            ents = []
            for start, end, label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode='contract')
                if span is None:
                    pass
                else:
                    ents.append(span)
            doc.ents = filter_spans(ents)
            doc_bin.add(doc)
        path = os.path.join(path, file_name[:-4])
        doc_bin.to_disk(f'{path}.spacy')
        return doc_bin

    def load_doc_bin(self, combined_name: str) -> DocBin:
        """
        Loads a Spacy doc_bin object from disk and returns it.
        """

        doc_bin = DocBin().from_disk(f'{combined_name[:-4]}.spacy')
        return doc_bin

    def generate_spacy_output(self, doc_bin_path, output_path: str, config_path: str) -> str:
        """
        Generates a Spacy training command and writes it to a txt file.
        Returns the Spacy training command.
        """

        with open(doc_bin_path, 'w') as f:
            spacy_output = f'spacy train {config_path} --output {output_path} --paths.train {doc_bin_path} --paths.dev {doc_bin_path}'
            f.write(spacy_output)
            f.close()
        return spacy_output




if __name__ == '__main__':
    pass