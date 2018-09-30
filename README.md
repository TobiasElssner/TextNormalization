# TextNormalization
The files are organized as follows:
- glove.*.txt : Embeddings for Twitter and canonical data (from SharedTask 2015) can be found here: 
                https://noisy-text.github.io/norm-shared-task
- lookup_ed.p: Lookup dictionary from unnormalized to normalized forms for embedding dimension e
- test_data_ne.json: Results for test_data.json for n-gram of order n and embedding size e
- test_truth.json: Gold standard annotations for the normalizations (from SharedTask2015)
- spoken.txt: Background corpus for the n-gram statistics
- word_list.p: List of all tokens occurring in spoken.txt
- ExtractNgrams.py: Extraxts ngrams from the background corpus\\
  Sequence counts for the kneser-ney-smoothing can be obtained there
- Lookup.py: Creates the dictionary from unnormalized to normalized forms
- Normalization.py: Normalizes data in json-format
- evaluation.py: Evaluates the results coming from Normalization.py (from SharedTask 2015)
