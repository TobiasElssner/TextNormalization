# TextNormalization
The files are organized as follows:
- glove.* : Embedding data (from SharedTask 2015)
- kneser_ney_n.p: Sequence counts for n-gram models
- lookup_ed.p: Lookup dictionary from unnormalized to normalized forms for embedding dimension e
- test_data_ne.json: Results for test_data.json for n-gram of order n and embedding size e
- test_truth.json: Gold standard annotations for the normalizations (from SharedTask2015)
- spoken.txt: Background corpus for the n-gram statistics
- word_list.p: List of all tokens occurring in spoken.txt
- ExtractNgrams.py: Extraxts ngrams from the background corpus
- Lookup.py: Creates the dictionary from unnormalized to normalized forms
- Normalization.py: Normalizes data in json-format
- evaluation.py: Evaluates the results coming from Normalization.py (from SharedTask 2015)
