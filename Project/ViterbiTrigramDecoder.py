from collections import defaultdict
import math
import sys
import numpy as np
import codecs
np.set_printoptions(threshold=sys.maxsize)
#python ViterbiTrigramDecoder.py --probs trigram_probs.txt --file mistyped_test.txt --check
"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class ViterbiTrigramDecoder(object):
    """
    This class implements Viterbi decoding using trigram probabilities in order
    to correct keystroke errors.
    """
    def init_a(self, filename):
        """
        Reads the trigram probabilities (the 'A' matrix) from a file.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                if len(line.split()) == 4 and line.split()[1].isnumeric():
                    i, j, k, d = [func(x) for func, x in zip([int, int, int, float], line.strip().split(' '))]
                    self.a[i][j][k] = d
        self.a = list(self.a.items())


    # ------------------------------------------------------


    def init_b(self, translation_list, translation):
        """
        Initializes the observation probabilities (the 'B' matrix).
        """
        for i in range(len(translation_list)):
            for j in range((len(translation[translation_list[i]]))):
                self.b[i][j] = math.log(0.1)  # Constant value

        """
        for i in range(max_translations):
            cs = Key.neighbour[i]

            # Initialize all log-probabilities to some small value.
            for j in range(max_translations):
                self.b[i][j] = -float("inf")

            # All neighbouring keys are assigned the probability 0.1
            for j in range(len(cs)):
                self.b[i][Key.char_to_index(cs[j])] = math.log( 0.1 )

            # The remainder of the probability mass is given to the correct key.
            self.b[i][i] = np.log((10 - len(cs))/10.0)"""


    # ------------------------------------------------------



    def viterbi(self, translation_list, max_translations, translations):
        """
        Performs the Viterbi decoding and returns the most likely
        string.
        """
        start_end = max_translations-1
        # First turn chars to integers, so that 'a' is represented by 0, 
        # 'b' by 1, and so on.
        # index = [Key.char_to_index(x) for x in s]

        # The Viterbi matrices
        self.v = np.zeros((len(translation_list), max_translations, max_translations), dtype='double')
        self.v[:,:,:] = -float("inf")
        self.backptr = np.zeros((len(translation_list), max_translations, max_translations), dtype='int')

        # Initialization
        self.backptr[0,:,:] = start_end
        self.v[0,:,:] =  np.add(self.a[start_end,:,:], self.b[0,:])
        self.v[1,:,:] =  np.add(self.a[start_end, start_end, :], self.b[0,:])
        self.backptr[1,:,:] = start_end

       
        for t in range(2, len(translation_list)):
            k = translation_list[t]
            for j in range(max_translations):
                D = np.add(self.v[t-1, j], self.a[:, j, k])
                self.v[t, k, j] = max(D) + self.b[k,k]
                self.backptr[t, k, j] = list(D).index(max(D))

                for translation in range(translations[k]):                
                    D = np.add(self.v[t-1, j], self.a[:, j, translation])     
                    self.v[t, translation, j] = max(D) + self.b[k, translation]
                    self.backptr[t, translation, j] = list(D).index(max(D))
        
        
        print_list = [start_end]  # Startar med start/slut
        index_1_step = start_end
        
        for i in range(len(translation_list)-1, 0, -1):
            index_2_step = self.backptr[i, print_list[len(print_list)-1], index_1_step]
            print_list.append(index_1_step)
            index_1_step = index_2_step

        print_list.reverse()
        result = []
        m = 0
        for j in print_list[:-2]:
            result = result + translations[translation_list[m]][j]
            m+=1
        return result

    # ------------------------------------------------------

    def __init__(self, length, max_translations, number_of_words):
        """
        Constructor: Initializes the A and B matrices.
        """

        # The trellis used for Viterbi decoding. The first index is the time step.
        self.v = None

        # The trigram stats.
        #self.a = np.zeros((length, number_of_words, number_of_words), dtype='float')
        self.a = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # The observation matrix.
        self.b = np.zeros((length, max_translations), dtype='double')

        # Pointers to retrieve the topmost hypothesis.
        self.backptr = None

    # ------------------------------------------------------