#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf8

import json
import numpy as np
import math
import codecs
import pickle
"""
Normalizes a given json file
"""


class Normalization:

    lookup = {}

    ngram_counts = {}
    word_list = {}

    n = 0

    # Kneser-Ney-constants
    D1 = 0
    D2 = 0
    D3 = 0

    # Numeral to character replacement
    num_to_letter = {"0": "O", "1": "o", "2": "t", "3": "t", "4": "f", "5": "f", "6": "s", "7": "s", "8": "e", "9": "n"}
    to_be_normalized = ""
    normalized = ""

    def __init__(self, ngram_counts, word_list, lookup, n, to_be_normalized, normalized):

        """
        Initialization
        :param ngram_counts: path to the dictionary with the raw sequence counts
        :param word_list: path to the list of all words
        :param lookup: path to the look-up from unnormalized to normalized forms
        :param n: order of the n-gram model
        :param to_be_normalized: path to the json-file with unnormalized data
        :param normalized: path to the destination of the json-file with normalized data
        """
        self.ngram_counts = pickle.load(open(ngram_counts, 'rb'))
        print("Ngrams read...")

        self.word_list = pickle.load(open(word_list, 'rb'))
        print("...word list read...")

        self.lookup = pickle.load(open(lookup, 'rb'))
        print("...lookup read.")
        self.n = n
        self.to_be_normalized = to_be_normalized
        self.normalized = normalized
        print("\n")

        self.initialize_kn_constants()
        self.normalize()

    def initialize_kn_constants(self):
        """
        Initializes the constants for Modified Kneser-Ney-Smoothing following [Chen and Goodman, 1999]
        """
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0

        for word in self.ngram_counts:

            for history in self.ngram_counts[word]:

                if len(history.split(" ")) == (self.n - 1):
                    if self.ngram_counts[word][history] == 1.0:
                        n1 += 1.0
                    elif self.ngram_counts[word][history] == 2.0:
                        n2 += 1.0
                    elif self.ngram_counts[word][history] == 3.0:
                        n3 += 1.0
                    elif self.ngram_counts[word][history] == 4.0:
                        n4 += 1.0

        y = n1 / (n1 + 2 * n2)

        self.D1 = 1 - (2 * y * n2 / n1)
        self.D1 = 2 - (3 * y * n3 / n2)
        self.D3 = 3 - (4 * y * n4 / n3)

    def pkn(self, current_word, history):

        """
        Computes recursively the probability for a word given a history following [Chen and Goodman]
        :param current_word: the word for which the probability is calculated
        :param history: the sequence of words preceding current_word
        :return: the probability of current_word given history
        """

        enumerator = 0.0
        if current_word in self.ngram_counts:
            if history in self.ngram_counts[current_word]:

                enumerator = self.ngram_counts[current_word][history]

                if enumerator == 1.0:
                    enumerator -= self.D1
                elif enumerator == 2.0:
                    enumerator -= self.D2
                elif enumerator >= 3.0:
                    enumerator -= self.D3

        denominator = 0.0

        n1 = 0.0
        n2 = 0.0
        n3 = 0.0

        for word in self.ngram_counts:

            if history in self.ngram_counts[word]:
                denominator += self.ngram_counts[word][history]

                if self.ngram_counts[word][history] == 1.0:
                    n1 += 1
                elif self.ngram_counts[word][history] == 2.0:
                    n2 += 1
                elif self.ngram_counts[word][history] >= 3.0:
                    n3 += 1

        if denominator != 0.0:
            gamma = (self.D1 * n1 + self.D2 * n2 + self.D3 * n3) / denominator

            if history != "" and history != " ":
                return (enumerator / denominator) + gamma * self.pkn(current_word, ' '.join(history.split(" ")[1:]))

            else:
                return (enumerator / denominator) + gamma
        else:
            return 0

    def normalize(self):

        """
        Normalizes the input sentences in a json file
        :return:
        """

        data = json.load(open(self.to_be_normalized))

        for elem in data:

            unnormalized_text = [x.lower().replace(" ", "").replace("\t", "").replace("-", " -").replace(",", " ,")
                                     .replace(".", " .").replace(";", " ;").replace("?", " ?").replace("(", "( ")
                                     .replace(")", " )").replace("{", "{ ").replace("}", " }")
                                 for x in elem["input"]]
            history = []
            start = "START"

            for i in range(self.n):
                history.insert(0, start + str(self.n - i))

            normalized_text = []

            for word in unnormalized_text:

                # If the word is not a user name
                if not (word.startswith("@") or word.startswith("#")):

                    multiword = []
                    multiword_prob = 1.0

                    # If the token has four or less characters, it could be a multi-word
                    if len(word) <= 4:

                        alt_history = history

                        # Each letter is treated as the first letter of another word it could stand for
                        for char in word:

                            # Numbers are replaced by the first character of their orthographic string
                            if char.isdigit():
                                letter = self.num_to_letter[char]

                                max_prob = 0.0
                                max_prob_word = ""


                                if letter in self.word_list:
                                    for possible_word in self.word_list[letter]:
                                        prob = self.pkn(possible_word, ' '.join(alt_history[:]))

                                        if prob > max_prob:
                                            max_prob = prob
                                            max_prob_word = possible_word

                                # If no word starts with the character - e.g. in case the 'letter' is a hyphen -
                                # All words are taken into account
                                else:
                                    for key in self.word_list:
                                        for possible_word in self.word_list[key]:
                                            prob = self.pkn(possible_word, ' '.join(alt_history[:]))

                                            if prob > max_prob:
                                                max_prob = prob
                                                max_prob_word = possible_word

                                multiword_prob *= max_prob
                                multiword.append(max_prob_word)

                                alt_history = alt_history[1:]
                                alt_history.append(max_prob_word)

                    one_word_prob = self.pkn(word, ' '.join(history[:]))
                    one_word = word

                    candidate = ""
                    candidate_prob = 0.0

                    # If the word occurs in the lookup, compute the probability for all of its normalized candidates
                    if word in self.lookup:

                        candidates = self.lookup[word]

                        for (can, sim) in candidates:

                            if can:
                                prob = self.pkn(can, ' '.join(history[:]))

                                if prob > candidate_prob:
                                    candidate_prob = prob
                                    candidate = can

                    if not multiword:
                        multiword_prob = 0.0

                    # Compare the probability for the recovered ,ost-probably multi-word phrase, the unnormalized token,
                    # And the most probably neighbour for the current word
                    # Take the most-probable normalization
                    largest = max(one_word_prob, multiword_prob, candidate_prob)

                    if largest == one_word_prob:
                        normalized_text.append(one_word)
                        history.append(word)
                        history = history[1:]

                    elif largest == multiword_prob:
                        normalized_text.extend(multiword)
                        history.extend(multiword)
                        history = history[len(word):]

                    else:
                        normalized_text.append(candidate)
                        history.append(candidate)
                        history = history[1:]

                # Append the normalization to the normalized text
                else:
                    normalized_text.append(word)

            # Save it in output
            elem["output"] = normalized_text

            # Store the json-file with the normalized texts
            with open(self.normalized, 'w') as outfile:
                json.dump(data, outfile)

Normalization("/path/to/n-gram-counts.p",
              "/path/to/word_list.p",
              "/path/to/lookup.p", n,
              "/path/to/test_data.json",
              "/path/to/normalized_destination.json")
