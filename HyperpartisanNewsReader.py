##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from html import unescape
from scipy import sparse
from lxml import etree
from collections import Counter
import sys

#####################################################################
# HELPER FUNCTIONS
#####################################################################
'''Loads the file in iteratively using a lazy list of elements in the XML tree
   Clears each element after reading its information so that memory doesn't get filled up
   This method acts as a generator, yielding elements on request instead of all at once.
   islice takes the iterable given from etree.iterparse and returns an iterator which will yield the next element upon request (up to max_elements)
'''
def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """
    Parses cleaned up spacy-processed XML files
    """
    fp.seek(0)

    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        yield elem
        elem.clear()
        if progress_message and (i % 1000 == 0):
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

'''Dumb because it loads the whole file into memory, then processes it all, and then takes a suitable slice'''
def dumb_xml_parse(fp, tag, max_elements=None):
    """
    Parses cleaned up spacy-processed XML files (but not very well)
    """
    elements = etree.parse(fp).findall(tag)
    N = max_elements if max_elements is not None else len(elements)
    return all_elems[:N]

#############################[1 for article in do_xml_parse(data_file, 'article')]########################################
# HNVocab
#####################################################################

class HNVocab(object):
    '''Takes a vocab file which has stop words at the start and then the text
       Lazily slices through the text to get each word in a specified range
       Makes a data structure out of these and bookkeeps'''
    def __init__(self, vocab_file, vocab_size, num_stop_words):
        start_index = 0 if num_stop_words is None else num_stop_words
        end_index = start_index + vocab_size if vocab_size is not None else None

        # self._stops = [w.strip() for w in islice(vocab_file, 0, start_index)]
        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)]
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)])

    '''Magic method, gives the number of words in the dictionary'''
    def __len__(self):
        return len(self._dict)

    '''Returns the word at a given index'''
    def index_to_label(self, i):
        return self._words[i]

    '''Magic method, gets the item associated with a key'''
    def __getitem__(self, key):
        if key in self._dict: return self._dict[key]
        else: return None


#####################################################################
# HNLabels
#####################################################################

class HNLabels(ABC):
    def __init__(self):
        self.labels = None
        self._label_list = None

    """ Magic method that gets the label in _label_list at a given index"""
    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    """ Given a label_file and an optional max_instances, parses the label_file
        and then extracts the labels for each article. Stores the labels in _label_list
        and a dictionary of label: index pairs in self.labels. Returns a list of the
        indexes of the extracted labels for each article."""
    def process(self, label_file, max_instances=None):
        articles = do_xml_parse(label_file, 'article', max_elements=max_instances)
        y_labeled = list(map(self._extract_label, articles))
        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])

        y = [self.labels[x] for x in y_labeled]
        return y

    """ Declares the _extract_label method which will be implemented in our subclass"""
    @abstractmethod
    def _extract_label(self, article):
        """ Return the label for this article """
        return "Unknown"

#####################################################################
# HNFeatures
#####################################################################

class HNFeatures(ABC):
    def __init__(self, vocab):
        self.vocab = vocab

    """ Extracts the text from a given article and cleans and lowercases it"""
    def extract_text(self, article):
        return unescape("".join([x for x in article.find("spacy").itertext()]).lower()).split()

    """ Given a data_file and an optional max_instances returns a matrix that has all the features
        for each article, and a list containing the ids of each article."""
    def process(self, data_file, max_instances=None):
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances

        X = sparse.lil_matrix((N, self._get_num_features()), dtype='float64')

        ids = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for i, article in enumerate(articles):
            ids.append(article.get("id"))
            for j, value in self._extract_features(article):
                X[i,j] = value
        return X, ids

    """ Abstract method to be implemented by our subclass"""
    @abstractmethod
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    """ Abstract method to be implemented by our subclass"""
    @abstractmethod
    def _extract_features(self, article):
        """ Returns a list of the features in the article """
        return []

    """ Abstract method to be implemented by our subclass"""
    @abstractmethod
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################
class BinaryLabel(HNLabels):
    def _extract_label(self, article):
        return article.get('hyperpartisan')

#####################################################################

class BagOfWordsFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return self.vocab[i]

    def _extract_features(self, article):
        """ Returns a list of the features in the article """
        text = self.extract_text(article)
        c = Counter()
        for word in text:
            if self.vocab[word] != None:
                c[word] += 1

        return [(self.vocab[word], count) for word, count in c.items()]

    def _get_num_features(self):
        """ Return the total number of features """
        return len(self.vocab)
