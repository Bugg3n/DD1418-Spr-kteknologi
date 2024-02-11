import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""
'''
This file is modified by Hugo Westergård and William Bork 2022 to be able to read the trigram-probabilities aswell. It was orginally called generator.py.
'''

class Reader(object): # Orginally Generator.
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram probabilities.
        self.bigram_prob = defaultdict(dict)

        # The trigram probabilities
        self.trigram_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            f = open(filename, 'r')
            self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
            for i in range (self.unique_words):
                index, word, count = map(str, f.readline().strip().split(' '))
                self.index[word] = index
                self.word[index] = word
                self.unigram_count[word] = count
            while True:
                try:
                    index1, index2, probability = map(str, f.readline().strip().split(' '))
                    self.bigram_prob[self.word[index1]][self.word[index2]] = math.exp(float(probability))
                except ValueError:
                    break
            while True:
                try:
                    index1, index2, index3, probability = map(str, f.readline().strip().split(' '))
                    self.trigram_prob[self.word[index1]][self.word[index2]][self.word[index3]] = math.exp(float(probability))
                except ValueError:
                    break
            return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

def main():
    reader = Reader()
    reader.read_model("trigram_probs.txt")

if __name__ == "__main__":
    main()