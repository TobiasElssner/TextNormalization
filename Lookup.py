#!/usr/bin/env python3
import numpy as np
import math
import pickle

"""
Creates a look-up dictionary from unnormalized to
normalized forms following [Sridhar 2015]
"""


class Lookup:

    vec_length = 0

    canonical_vecs = {}
    unnormalized_vecs = {}

    lookup = {}

    def __init__(self, dimensions, canonical, unnormalized):

        """
        Initialization
        :param dimensions: dimension of the embeddings
        :param canonical: embeddings for canonical data
        :param unnormalized: embeddings for unnormalized data
        """

        self.vec_length = dimensions
        self.read_vecs(canonical, unnormalized)
        self.create_lookup()

    def read_vecs(self, canonical, unnormalized):

        """
        Reads both canonical and unnormalized embeddings and stores them in dictionaries.
        In this general Version of the code, the number of canonical and unnormalized tokens is not artificially reduced
        :param canonical: path to canonical embeddings
        :param unnormalized: path to unnormalized embeddings
        """

        with open(canonical) as f:
            for line in f:

                tmp_vec = np.zeros((self.vec_length, 1))

                split = line.split(" ")
                word = split[0]

                for i in range(1, len(split)):
                    tmp_vec[i - 1] = float(split[i].strip("\n"))
                self.canonical_vecs[word] = tmp_vec

        with open(unnormalized) as f:
            for line in f:

                tmp_vec = np.zeros((self.vec_length, 1))

                split = line.split(" ")
                word = split[0]

                for i in range(1, len(split)):
                    tmp_vec[i - 1] = float(split[i].strip("\n"))
                self.unnormalized_vecs[word] = tmp_vec

    def cos_sim(self, vec1, vec2):

        """
        Calculates the cosine distance between two vectors
        :param vec1: first vector
        :param vec2: second vector
        :return: cosine distance
        """

        numerator = 0.0
        denominator = 0.0

        for i in range(self.vec_length):
            numerator += vec1[i] * vec2[i]
            denominator += ((vec1[i] * vec1[i]) + (vec2[i] * vec2[i]))

        denominator = math.sqrt(denominator)

        return numerator / denominator

    def get_top_25(self, canonical_word):
        """
        Finds the top 25 most similar 'unnormalized' vectors for a canonical one
        :param canonical_word: the word for which the top 25 nearest neighbours are calculated
        :return: a list of 25 most similar unnormalized words
        """

        canonical_vec = self.unnormalized_vecs[canonical_word]

        top_list = [("", -1)] * 25

        for unnormalized_word in self.unnormalized_vecs:

            sim = self.cos_sim(canonical_vec, self.unnormalized_vecs[unnormalized_word])

            index = 24

            while top_list[index][1] > sim and index > 0:
                index -= 1

            if top_list[index][1] > sim:
                top_list.insert(index, (unnormalized_word, sim))
            else:
                top_list.insert(index + 1, (unnormalized_word, sim))

            del top_list[-1]

        return [x[0] for x in top_list]

    def lex_sim(self, word1, word2):
        """
        Computes the lexical similarity between two words following [Sridhar 2015]
        :param word1: first word
        :param word2: second word
        :return: lexical similarity between word1 and word2
        """
        return self.lcsr(word1, word2) / self.levenshtein_distance(word1, word2)

    def lcsr(self, word1, word2):

        """
        Calculates the longest common substring ratio between two words, i.e. their longest common substring divided by
        the length of the longer word
        :param word1: first word
        :param word2: second word
        :return: longest common substring ratio between word1 and word2
        """

        max_length = len(word1)

        if len(word2) > max_length:
            max_length = len(word2)

        return self.longest_common_substring(word1, word2) / max_length

    def longest_common_substring(self, word1, word2):
        """
        Calculates the longest common substring between two words
        :param word1: first word
        :param word2: second word
        :return: longest common substring between word1 and word2
        """
        lcs = 0

        for i in range(len(word1)):
            for j in range(i + 1, len(word1)):

                substring = ''.join(word1[i:j])

                if substring in word2 and len(substring) > lcs:
                    lcs = len(substring)
        return lcs

    def levenshtein_distance(self, word1, word2):

        """
        Measures Levenshtein-Distance between two words
        :param word1: first word
        :param word2: second word
        :return: Levenshtein-Distance between word1 and word2
        """

        word1 = self.replace_vowels(word1)
        word2 = self.replace_vowels(word2)

        word1_length = len(word1)
        word2_length = len(word2)

        m = np.zeros((len(word1) + 1, len(word2) + 1))
        m[0][0] = 0

        for i in range(word1_length + 1):
            for j in range(word2_length + 1):

                if i == 0:
                    m[i][j] = j
                if j == 0:
                    m[i][j] = i

                if i > 0 and j > 0:

                    substitution = 0

                    if word1[i - 1] != word2[j - 1]:
                        substitution = 1

                    m[i][j] = min(m[i - 1][j] + 1,
                                  m[i][j - 1] + 1,
                                  m[i - 1][j - 1] + substitution)

        return m[word1_length][word2_length]

    def replace_vowels(self, word):

        """
        Reduces a word to its consonant skeleton
        :param word: the string of the word to be reduced
        :return: the consonant skeleton of word
        """

        word = word.replace("a", "")
        word = word.replace("e", "")
        word = word.replace("i", "")
        word = word.replace("o", "")
        word = word.replace("u", "")
        word = word.replace("y", "")

        return word

    def create_lookup(self):

        """
        Creates the look-up for unnormalized forms by inverting the existing dictionary from normalized forms to their
        top-25 unnormalized neighbours.
        Adds also the lexical similarity for all neighbours and sorts them accordingly.
        """

        lex_sims = []

        for canonical_word in self.canonical_vecs:

            top_25_neighbours = self.get_top_25(canonical_word)

            for neighbour in top_25_neighbours:

                top_list = [("", -1)] * 25

                if neighbour in self.lookup:
                    top_list = self.lookup[neighbour]

                # add the lexical similarity to each of the 25 neighbours and re-sort them accordingly
                sim = self.lex_sim(canonical_word, neighbour)
                lex_sims.append(sim)

                index = 24

                while top_list[index][1] > sim and index > 0:
                    index -= 1

                if top_list[index][1] > sim:
                    top_list.insert(index, (canonical_word, sim))
                else:
                    top_list.insert(index + 1, (canonical_word, sim))

                del top_list[-1]

                self.lookup[neighbour] = top_list

Lookup(dimension, "/path/to/normalized_embeddings.txt", "/path/to/unnormalized_embeddings.txt")
