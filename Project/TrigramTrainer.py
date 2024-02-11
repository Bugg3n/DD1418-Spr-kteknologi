#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""
"""
Modified by William Bork and Hugo Westergård in 2022. Original version named "BigramTrainer" has been extended and 
re-named "TrigramTrainer".
"""

class TrigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """
    
    def clean_word(self, token):
        new_word_list = []
        word_list = [*token]
        for i in word_list:
            if i == "/":
                break
            else:
                new_word_list.append(i)
        self.process_token("".join(new_word_list))

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        self.total_words += 1
        token = self.removePunctuation(token)
        if self.unigram_count[token] == 0:
            self.index[token] = self.unique_words
            self.word[self.unique_words] = token
            self.unique_words += 1
        self.unigram_count[token] += 1

        if self.second_last_index != -1:
            self.trigram_count[self.second_last_index][self.last_index][token] += 1
  
        if self.last_index != -1:
            self.bigram_count[self.last_index][token] += 1

        self.second_last_index = self.last_index
        self.last_index = token
        
    def removePunctuation(self, word):
        newWord = []
        word_list = [*word]
        for i in word_list:
            if i.isalpha():
                newWord.append(i)
        return "".join(newWord).lower()

    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """

        rows_to_print = []

        rows_to_print.append(str(self.unique_words) + " " + str(self.total_words))
        
        for i in range(self.unique_words):
            rows_to_print.append(str(i) + " " + self.word[i] + " " + str(self.unigram_count[self.word[i]]))
        for j in self.index:
            for k in self.bigram_count[j]:
                rows_to_print.append(str(self.index[j]) + " " + str(self.index[k]) + " " + str(math.log((self.bigram_count[j][k]/self.unigram_count[j]))))
        rows_to_print.append("-1")
        for l in self.index:
            for m in self.trigram_count[l]:
                for n in self.trigram_count[l][m]:
                    rows_to_print.append(str(self.index[l]) + " " + str(self.index[m]) + " " + str(self.index[n]) + " " + str(math.log((self.trigram_count[l][m][n]/self.unigram_count[l]))))

        rows_to_print.append("-1")
        
        return rows_to_print

    def loopFiles(self, filename):
        path = os.getcwd()
        path = os.path.join(path, filename)
        for file in os.listdir(path):
            if file == ".DS_Store":
                pass
            else:
                f = os.path.join(path, file)
                print(file)
                with open(f) as file:
                    for row in file:
                        for word in row.split():
                            self.clean_word(word)

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # The identifier of the previous word processed.
        self.last_index = -1

        self.second_last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = TrigramTrainer()

    bigram_trainer.loopFiles(arguments.file)

   
    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)

if __name__ == "__main__":
    main()
