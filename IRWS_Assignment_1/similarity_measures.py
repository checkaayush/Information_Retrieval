"""Script to compare different similarity measures

Attributes:
    NUM_DOCS (int): Number of documents to be downloaded
"""

import subprocess
import urllib
import glob
import numpy
from math import log10

from nltk.corpus import stopwords
from google import search

NUM_DOCS = 2


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
        # Hacky check to get 10 docs
        global NUM_DOCS
        if doc_count == NUM_DOCS:
            break


def doc_to_text_catdoc(filename):
    """Convert .doc file to .txt file.

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


def remove_stopwords(filename):
    """Remove stopwords from file.
       Creates new file without stopwords with prefix - 'processed'.

    Args:
        filename (string): Name of text file to be processed

    """
    word_list = open(filename, "r")
    stops = set(stopwords.words('english'))
    processed_file = "processed_" + filename.split('.')[0] + ".txt"

    file_string = ""
    for line in word_list:
        for w in line.split():
            if w.lower() not in stops:
                # print w
                file_string = file_string + " " + w

    with open(processed_file, "w") as f:
        f.write(file_string)


def preprocess(filename):
    """Preprocess the raw text file.

    Args:
        filename (string)
    """
    remove_stopwords(filename)


def get_termfreq_list(query, filename):
    """Calculate frequency of each query term in file

    Args:
        query (string): Search query
        filename (string)

    Returns:
        list: List of term frequencies in file
    """
    tf_list = []

    with open(filename, 'r') as f:
        raw_text = f.read()

    for term in query.split():
        tf = raw_text.lower().count(term)
        tf_list.append(tf)

    return tf_list


def generate_tf_idf_matrix(tf_matrix):
    """Generates TF-IDF weight matrix.

    Args:
        tf_matrix (numpy array): Term frequency matrix

    Returns:
        numpy array: Required weight matrix
    """
    (rows, cols) = tf_matrix.shape

    # Create weight matrix
    weight_matrix = numpy.zeros(shape=(rows, cols), dtype=numpy.float)

    for i in range(rows):
        for j in range(cols):
            tf = tf_matrix[i][j]

            # Calculate document frequency
            df = 0
            for entry in tf_matrix[:, j]:
                # print "entry ", entry
                if entry > 0:
                    df += 1

            idf = 1 + log10((10 * NUM_DOCS) / df)

            weight_matrix[i][j] = tf * idf

    return weight_matrix


def main():

    # Default query for testing
    # query = "anna hazare"

    query = raw_input("Query: ")
    query = query.lower()
    # download_documents(query)

    # Generate text files from doc files
    # doc_files = glob.glob('*.doc')
    # if doc_files:
    #     for f in doc_files:
    #         doc_to_text_catdoc(f)
    # else:
    #     print "No doc files to process."

    # Preprocess text files
    # text_files = glob.glob('*.txt')
    # if text_files:
    #     for f in text_files:
    #         preprocess(f)
    # else:
    #     print "No text files to process."

    # Generate Frequency Matrix
    processed_files = glob.glob('processed*.txt')
    num_files = len(processed_files)
    num_terms = len(query.split())
    tf_matrix = numpy.zeros(shape=(num_files, num_terms))

    if processed_files:
        index = 0
        for f in processed_files:
            print "\nProcessed file : %s" % f
            tf_matrix[index] = get_termfreq_list(query, f)
            index += 1
    else:
        print "No files to process."
    print "\nTerm frequency matrix:\n\n", tf_matrix

    # Calculate TF-IDF weighting
    weight_matrix = generate_tf_idf_matrix(tf_matrix)
    print "\nTF-IDF Weight Matrix:\n\n", weight_matrix

if __name__ == "__main__":
    main()
