"""Script to compare different similarity measures

Attributes:
    NUM_DOCS_DOWNLOAD (int): Number of documents to be downloaded
    TOTAL_DOCS (int): Total number of documents in the corpus
"""

import sys
import subprocess
import urllib
import glob
import numpy
import math
from collections import OrderedDict

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer
from google import search

import similarity_measures

TOTAL_DOCS = 3
NUM_DOCS_DOWNLOAD = 1

# Set default character encoding
reload(sys)
sys.setdefaultencoding('UTF8')


def download_documents(query):
    """Download .doc files for given search query from Google

    Args:
        query (string): Search query

    """

    # Modified Google search query for .doc files
    query += " filetype:doc"

    doc_count = 0
    for url in search(query, stop=50):

        # Cleaning the encoded URL for filename
        filename = urllib.unquote(url).split('/')[-1]

        print "\nDownloading: \nFilename: %s\nURL: %s" % (filename, url)
        urllib.urlretrieve(url, filename)
        # print urllib.unquote(url)

        doc_count += 1
        # Hacky check to get desired number of docs
        global NUM_DOCS_DOWNLOAD
        if doc_count == NUM_DOCS_DOWNLOAD:
            break


def doc_to_text():
    """Generate text files from doc files."""

    doc_files = sorted(glob.glob('*.doc'))
    if doc_files:
        for f in doc_files:
            helper_doc_to_text(f)
    else:
        print "No doc files to process."


def helper_doc_to_text(filename):
    """Convert .doc file to .txt file using 'catdoc' linux utility

    Args:
        filename (string): Name of the file

    Raises:
        OSError
    """

    text_file = filename.split('.')[0] + ".txt"
    with open(text_file, "w") as fo:
        try:
            command = "catdoc"
            option = "-w"  # Disables word wrapping

            raw_text = subprocess.check_output([command, option, filename])

            fo.write(raw_text)
        except OSError:
            print "Command %s could not be executed." % command


def remove_stopwords(token_list):
    """Remove stopwords from list of words.

    Args:
        token_list (list): List of words including stopwords

    Returns:
        list: List of words without stopwords
    """

    stops = set(stopwords.words('english'))
    word_list = []

    for token in token_list:
        if token not in stops:
            word_list.append(token)

    return word_list


def preprocess_text(text):
    """Preprocess the text by Tokenization, Stopword Removal, etc.

    Args:
        text (string): Raw text to be processed

    Returns:
        list: List of processed words from text
    """

    # Character encoding and decoding
    text = text.decode(encoding='UTF-8', errors='ignore')
    text = text.encode(encoding='ASCII', errors='ignore')

    # Case folding to lowercase
    text = text.lower()

    # Tokenization
    token_list = word_tokenize(text)

    # Stopword-Removal
    word_list = remove_stopwords(token_list)

    # Stemming and/or Lemmatization (Reducing inflectional forms)
    # TODO: Might have to stem and/or lemmatize query as well
    # porter_stemmer = PorterStemmer()
    # word_list = [porter_stemmer.stem(word) for word in word_list]

    return word_list


def get_termfreq_list(query_terms, word_list):
    """Get term-frequency list

    Args:
        query_terms (list): List of query terms
        word_list (TYPE): List of words in processed document

    Returns:
        list: List of frequencies of query terms in documents
    """

    tf_list = []

    for term in query_terms:
        tf = word_list.count(term)
        tf_list.append(tf)

    return tf_list


def generate_tf_matrix(index_terms):
    """Generates Term-Frequency Matrix

    Args:
        index_terms (list): List of processed query terms

    Returns:
        numpy array: Term-Frequency matrix
    """

    # TODO: Only retrieved files should be used here.
    text_files = sorted(glob.glob('*.txt'))
    # print text_files

    num_files = len(text_files)
    num_index_terms = len(index_terms)
    tf_matrix = numpy.zeros(shape=(num_files, num_index_terms))

    if text_files:
        for doc_index, filename in enumerate(text_files):
            with open(filename, 'r') as f:
                word_list = preprocess_text(f.read())
                tf_list = get_termfreq_list(index_terms, word_list)
                tf_matrix[doc_index] = numpy.array(tf_list)
    else:
        print "No text files to process."

    return tf_matrix


def get_total_wordcount(filename):
    """Count number of words in processed file.

    Args:
        filename (string)

    Returns:
        int: Wordcount
    """

    with open(filename, 'r') as f:
        word_list = preprocess_text(f.read())
        return len(word_list)


def get_wordcount_list():
    """Get wordcounts for documents.

    Returns:
        list: List of wordcounts for documents
    """

    text_files = sorted(glob.glob('*.txt'))
    wordcount_list = []
    if text_files:
        for file_index, f in enumerate(text_files):
            wordcount_list.append(get_total_wordcount(f))
    else:
        print "No doc files to process."

    return wordcount_list


def normalize(tf_matrix, wordcount_list):
    """Normalize effect of length of document on term-frequency

    Args:
        tf_matrix (numpy array): Term-frequency matrix for documents
        wordcount_list (list): List of wordcounts for documents

    Returns:
        numpy array: Normalized Term-frequency matrix
    """

    (rows, cols) = tf_matrix.shape
    norm_tf_matrix = numpy.zeros(shape=(rows, cols), dtype=numpy.float)
    for r in range(rows):
        for c in range(cols):
            if wordcount_list[r]:
                norm_tf_matrix[r][c] = (float(tf_matrix[r][c]) /
                                        wordcount_list[r])
            else:
                norm_tf_matrix = tf_matrix.astype(float)

    return norm_tf_matrix


def get_idf(tf_matrix, term_index):
    """Calculates Inverse document frequency for given term.

    Args:
        tf_matrix (nump array): Term frequency matrix
        term_index (int): Index of term in query (0-based indexing)

    Returns:
        float: Inverse Document Frequency
    """

    # Document Frequency (DF)
    df = 0
    for entry in tf_matrix[:, term_index]:
        if entry > 0:
            df += 1

    # print "df = ", df
    # Inverse Document Frequency (IDF)
    if df:
        idf = 1.0 + math.log10(float(TOTAL_DOCS) / df)
    else:
        idf = 0.0

    return idf


def generate_weight_matrix(tf_matrix):
    """Generates TF-IDF weight matrix.

    Args:
        tf_matrix (numpy array): Term frequency matrix

    Returns:
        numpy array: Required weight matrix
    """
    (rows, cols) = tf_matrix.shape

    # Create weight matrix
    weight_matrix = numpy.zeros(shape=(rows, cols), dtype=numpy.float)

    # Normalize effect of length of document on term frequency
    wordcount_list = get_wordcount_list()
    norm_tf_matrix = normalize(tf_matrix, wordcount_list)

    for i in range(rows):
        for j in range(cols):

            # Normalized Term Frequency
            norm_tf = norm_tf_matrix[i][j]

            # Inverse Document Frequency
            idf = get_idf(tf_matrix, j)

            # (Normalized TF)-IDF weight matrix
            weight_matrix[i][j] = norm_tf * idf

    return weight_matrix


def create_corpus():
    """Create document corpus from user queries."""

    num_queries = input("\nNumber of queries: ")

    global NUM_DOCS_DOWNLOAD
    NUM_DOCS_DOWNLOAD = input("Number of docs for each query: ")

    global TOTAL_DOCS
    TOTAL_DOCS = num_queries * NUM_DOCS_DOWNLOAD

    print "Total docs = %d" % TOTAL_DOCS

    for i in range(num_queries):
        query = raw_input("\nQuery: ")
        index_terms = preprocess_text(query)
        search_query = ' '.join(index_terms)

        # Download documents
        download_documents(search_query)

    # Generate text files from .doc files
    doc_to_text()


def get_query_vector(index_terms, tf_matrix):
    """Creates query vector of query-term-weights.

    Args:
        index_terms (list):
        tf_matrix (numpy array):

    Returns:
        list: Query vector
    """

    # q_len = len(index_terms)
    q_vector = []
    for t in index_terms:
        t_count = index_terms.count(t)
        q_vector.append(t_count)

    # for t in range(q_len):
    #     tf = index_terms.count(index_terms[t])
    #     norm_tf = (float(tf) / q_len)
    #     # print "\nnorm_tf = ", norm_tf

    #     idf = get_idf(tf_matrix, t)
    #     # print "\nidf = ", idf

    #     weight = norm_tf * idf
    #     # print "tf * idf = ", weight

    #     q_vector.append(weight)

    return q_vector


def ranking_tf_idf(documents, tf_matrix):

    # TF-IDF weighting (Term-Document Matrix)
    weight_matrix = generate_weight_matrix(tf_matrix)
    print "\nTF-IDF Weight Matrix:\n\n", weight_matrix

    (rows, cols) = weight_matrix.shape
    tf_idf_rank = {}
    for row in range(rows):
        doc_score = sum(weight_matrix[row])
        tf_idf_rank[documents[row]] = doc_score

    tf_idf_rank = OrderedDict(sorted(tf_idf_rank.items(), key=lambda t: t[1],
                              reverse=True))
    # print "\ntf_idf_ranks : ", tf_idf_rank
    return tf_idf_rank


def print_ranking(ranks_dict):
    print "\nRank\tFilename\tScore\n"
    for i, t in enumerate(ranks_dict):
        print "%d\t%s\t%f" % (i + 1, t, ranks_dict[t])
    print


def ranking_cos_sim(documents, index_terms):
    cos_sim_rank = {}
    for i, d in enumerate(documents):
        with open(d, 'r') as f:
            text = f.read()

        # Build vocabulary (exhaustive word set for doc + query)
        word_list = preprocess_text(text)
        result_list = [index_terms, word_list]
        vocabulary = set().union(*result_list)

        doc_vector = [0] * len(vocabulary)
        query_vector = [0] * len(vocabulary)
        for j, word in enumerate(vocabulary):
            count = word_list.count(word)
            doc_vector[j] = count
            if word in index_terms:
                # print word, count
                query_vector[j] = count

        # print doc_vector
        value = similarity_measures.cosine_similarity(query_vector,
                                                      doc_vector)
        # cos_sim.append(value)
        cos_sim_rank[d] = value
        cos_sim_rank = OrderedDict(sorted(cos_sim_rank.items(), key=lambda t:
                                   t[1], reverse=True))

    return cos_sim_rank


def main():

    # # TODO: Menu driven
    # # Create document corpus from user queries.
    # create_corpus()

    # New query
    new_query = raw_input("\nNew query: ")

    index_terms = preprocess_text(new_query)

    # # TODO: Generate TF matrix for only the relevant retrieved docs
    # Generate Term-Frequency matrix
    tf_matrix = generate_tf_matrix(index_terms)
    print "\nTerm Frequency Matrix:\n\n", tf_matrix

    documents = sorted(glob.glob('*.txt'))
    # print documents

    # TF-IDF Ranking
    tf_idf_rank = ranking_tf_idf(documents, tf_matrix)
    print "\nRanking based on TF-IDF Weighting: "
    print_ranking(tf_idf_rank)

    # Cosine Similarity based ranking
    cos_sim_rank = ranking_cos_sim(documents, index_terms)
    print "\nRanking based on Cosine Similarity: "
    print_ranking(cos_sim_rank)

    # Jaccard coefficient based ranking
if __name__ == "__main__":
    main()

# Some notes:
# Also, try Maximum-TF Normalization (dividing by max tf)?
# Something like: ntf = 0.4 + 0.6 * (tf / tf_max)
# ntf is also called Augmented term frequency.
# 0.4 is the smoothing term.
