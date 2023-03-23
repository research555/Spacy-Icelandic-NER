import os

import spacy
from spacy.scorer import Scorer
import pickle
from spacy.tokens import Doc, Span
from sklearn.metrics import classification_report
from pprint import PrettyPrinter
from tqdm import tqdm

class Test:
    """
       A class for evaluating a named entity recognition (NER) model and generating a report.

       Attributes:
           number (int): The number of the model to be evaluated.
           test_set_path (str): The file path of the test dataset to be used for evaluation.
               Default is None.

       Methods:
           evaluate_ner_model(): Loads the test dataset and evaluates the NER model. Calculates
               precision, recall, and F1 score for the model, as well as counts of correct and
               incorrect entities. Returns a report containing these metrics.
       """
    def __init__(self, number: int, test_set_path: str = None):
        self.model_path = rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\models\try{number}/model-best'
        self.test_set_path = test_set_path
        self.nlp = spacy.load(self.model_path)
        self.correct_ents = {
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
        self.incorrect_ents = {
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

    def evaluate_ner_model(self):
        """
                Initializes a new instance of the Test class.

                Parameters:
                    number (int): The number of the model to be evaluated.
                    test_set_path (str): The file path of the test dataset to be used for evaluation.
                        Default is None.
        """

        with open(self.test_set_path, 'rb') as f:
            data = pickle.load(f)

        incorrect_labels = []
        correct_labels = []
        tp, fp, fn = 0, 0, 0
        for example in tqdm(data.values()):
            doc = self.nlp(example['text'])
            true_labels = set([(start, end, label) for start, end, label in example['entities'] if label != 'O'])
            predicted_labels = set(
                [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ != 'O'])
            #print("Text:", example['text'])
            #print("Expected Entities:", true_labels)
            #print("Predicted Entities:", predicted_labels)
            for label in true_labels:
                if label in predicted_labels:
                    tp += 1
                    predicted_labels.remove(label)
                    self.correct_ents[label[2]] += 1
                else:
                    fn += 1
                    self.incorrect_ents[label[2]] += 1

            fp += len(predicted_labels)
        """for key1, label1 in self.incorrect_ents.items():
            for key2, label2 in self.correct_ents.items():
                if label2 == label1:
                    self.incorrect_ents[key1][label1] = (self.incorrect_ents[label1] * self.correct_ents[label2])/100

"""
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        output = f'model: {self.model_path}\ntest_set: {self.test_set_path}\n' \
                 f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}\n' \
                 f'Incorrect Entity Count: {self.incorrect_ents}\n' \
                 f'most inaccurate Ent: {max(self.incorrect_ents, key=self.incorrect_ents.get)}, ' \
                 f'number missed: {self.incorrect_ents[max(self.incorrect_ents, key=self.incorrect_ents.get)]}\n' \
                 f'least inaccurate Ent in percent missed: {min(self.incorrect_ents, key=self.incorrect_ents.get)}, ' \
                 f'number missed in percent missed: {self.incorrect_ents[min(self.incorrect_ents, key=self.incorrect_ents.get)]}\n' \
                 f'Correct Entity Count: {self.correct_ents}\n' \
                 f'Most Accurate Ent: {max(self.correct_ents, key=self.correct_ents.get)}, ' \
                 f'number correct: {self.correct_ents[max(self.correct_ents, key=self.correct_ents.get)]}\n' \
                 f'Least Accurate Ent: {min(self.correct_ents, key=self.correct_ents.get)}, ' \
                 f'number correct: {self.correct_ents[min(self.correct_ents, key=self.correct_ents.get)]}\n\n'

        with open('../predictions/predictions.txt', 'a') as f:
            f.write(output)
        return output, correct_labels, incorrect_labels


if __name__ == '__main__':

    test_paths = [#(10, rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits\train70test30\deleted90\TEST_combined_90_percent_70_30.pkl'),
                  #(11, rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits\train60test40\deleted60\TEST_combined_60_percent_60_40.pkl'),
                  (12, rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits\train70test30\deleted90\TEST_combined_90_percent_70_30.pkl'),
                  (13, rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits\train50test50\deleted50\TEST_combined_50_percent_50_50.pkl'),
                  (14, rf'C:\Users\imran\PycharmProjects\FactiVerse_Case\train_data\splits\train80test20\deleted90\TEST_combined_90_percent_80_20.pkl')]
    for item in test_paths:

        eval = Test(item[0], test_set_path=item[1])
        print(eval.evaluate_ner_model())
