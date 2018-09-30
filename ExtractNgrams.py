import pickle
import codecs

"""
Extracts ngrams from a given text and writes the raw counts in a dictionary and creates a second dictionary with words
as values and their first letters as keys
"""


class ExtractNgrams:
    corpus = ""
    ngram_dest = ""
    word_list_dest = ""
    n = 0

    ngram_counts = {}
    word_list = {}

    def __init__(self, corpus, ngram_dest, word_list_dest, n):
        """
        Initialization
        :param corpus: path to the corpus text file
        :param ngram_dest: destination for the dictionary
        :param word_list_dest: destination for the wordlist
        :param n: order of the n-gram
        """
        self.corpus = corpus
        self.ngram_dest = ngram_dest
        self.word_list_dest = word_list_dest
        self.n = n

        self.extract_ngrams()

    def extract_ngrams(self):

        """
        Loops over sentences in the corpus and extracts
        all sequences up to length n (necessary for Modified Kneser-Ney-Smoothing)
        """

        with open(self.corpus) as f:
            data = f.read().lower().replace("\t", "").\
                replace("-", " -").replace(",", " ,").replace(".", " .\n").replace(";", " ;").replace("?", " ?\n")

            sents = data.split("\n")

            for sent in sents:

                words = [x for x in sent.split(" ") if x]


                if len(words) != 0:

                    # Append n - 1 START symbols for the n-gram model
                    start = "START"

                    for i in range(self.n):
                        words.insert(0, start + str(self.n - i))

                    # Single END symbol is sufficient for this project
                    end = "END"

                    words.append(end)

                    for i in range(self.n, len(words)):

                        cur_word = words[i]

                        if cur_word:

                            first_letter = cur_word[0]

                            if first_letter in self.word_list:
                                if cur_word not in self.word_list[first_letter]:
                                    self.word_list[first_letter].append(cur_word)
                            else:
                                self.word_list[first_letter] = [cur_word]

                            # Include all preceding sequences of length < n in the dictionary
                            for j in range(self.n):
                                history = ' '.join(words[(i - j): i])

                                tmp_curword = {}

                                if cur_word in self.ngram_counts:
                                    tmp_curword = self.ngram_counts[cur_word]

                                tmp_curword.setdefault(history, 0.0)
                                tmp_curword[history] += 1.0
                                self.ngram_counts[cur_word] = tmp_curword

        # Save the results in a dictionary and a word list
        pickle.dump(self.ngram_counts, open(self.ngram_dest, 'wb'))
        pickle.dump(self.word_list, open(self.word_list_dest, "wb"))


ExtractNgrams("/path/to/corpus.txt",
              "/path/to/dictionary_destination.p",
              "/path/to/word_list_destination.p", n)

