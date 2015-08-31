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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer
from google import search

import similarity_measures

TOTAL_DOCS = 2
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

    # Tokenize
    token_list = word_tokenize(text)

    # Remove stopwords
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

    text_files = sorted(glob.glob('*.txt'))

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


def get_wordcount(filename):
    """Count number of words in processed file.

    Args:
        filename (string)

    Returns:
        int: Wordcount
    """

    with open(filename, 'r') as f:
        word_list = preprocess_text(f.read())
        return len(word_list)


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


def get_wordcount_list():
    """Get wordcounts for documents.

    Returns:
        list: List of wordcounts for documents
    """

    text_files = sorted(glob.glob('*.txt'))
    wordcount_list = []
    if text_files:
        for file_index, f in enumerate(text_files):
            wordcount_list.append(get_wordcount(f))
    else:
        print "No doc files to process."

    return wordcount_list


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

            # Document Frequency (DF)
            df = 0
            for entry in tf_matrix[:, j]:
                if entry > 0:
                    df += 1

            # Inverse Document Frequency (IDF)
            idf = 1 + math.log10(TOTAL_DOCS / df)

            # (Normalized TF)-IDF weight matrix
            weight_matrix[i][j] = norm_tf * idf

    return weight_matrix


def doc_to_text():
    """Generate text files from doc files."""

    doc_files = sorted(glob.glob('*.doc'))
    if doc_files:
        for f in doc_files:
            helper_doc_to_text(f)
    else:
        print "No doc files to process."


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


# def get_query_vector():


def main():

    # # Create document corpus from user queries.
    # create_corpus()

    # # New query
    new_query = raw_input("\nNew query: ")
    # len_new_query = len(new_query.split())

    modified_query = preprocess_text(new_query)
    index_terms = modified_query.split()

    # # Generate Term-Frequency matrix
    tf_matrix = generate_tf_matrix(index_terms)
    print "\nTerm Frequency Matrix:\n\n", tf_matrix

    # # Calculate TF-IDF weighting (Term-Document Matrix)
    weight_matrix = generate_weight_matrix(tf_matrix)
    print "\nTF-IDF Weight Matrix:\n\n", weight_matrix

    # query_vector = get_query_vector(len_new_query, index_terms)
    cos_sim = similarity_measures.cosine_similarity([1, 2, 3], [3, 2, 1])
    print cos_sim

    # Some notes:
    # 1) Document score will be addition of TF-IDF weighting scores
    # over all query terms
    #
    # 2) Normalization based on Document length should consider
    # only the terms in vocabulary as superset (unique terms in
    # bag of words representation of document).
    #
    # 3) Ideally, cos_sim(query, doc) should have both vector sizes
    # equal to the size of the vocabulary. But, since we are considering
    # TF-IDF weights as the vector values, it will turn out to be
    # computationally very expensive for all terms in vocabulary
    # (and inefficient, since we haven't yet used the inverted-index form).
    # Instead, we could use vector size equal to size of query. Additionally,
    # we could incorporate term-weights for query terms.

if __name__ == "__main__":
    main()
