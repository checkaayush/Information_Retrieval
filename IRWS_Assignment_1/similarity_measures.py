"""Script to compare different similarity measures

Attributes:
    NUM_DOCS (int): Number of documents to be downloaded
"""

import subprocess
import urllib
import glob

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

        print "Downloading: \nFilename: %s\nURL: %s" % (filename, url)
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


def generate_tf_matrix(query, filename):
    

def main():

    # query = raw_input("Query: ")
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

    processed_files = glob.glob('processed*.txt')
    if processed_files:
        for f in processed_files:
            generate_tf_matrix(query, f)
    else:
        print "No files to process."


if __name__ == "__main__":
    main()
