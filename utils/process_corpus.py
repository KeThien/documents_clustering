from __future__ import print_function
from math import nan
import os
import re
from dask import delayed

import nltk.corpus
from unidecode                        import unidecode
from nltk.tokenize                    import word_tokenize
from nltk                             import SnowballStemmer
from nltk.stem                        import WordNetLemmatizer

dir_path = os.path.dirname(os.path.realpath(__file__))

# removes a list of words (ie. stopwords) from a tokenized list.
def remove_words(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

# applies lemmatizer to a list of tokenized words
def apply_lemmatizer(listOfTokens, lemmatizer):
    return [lemmatizer.lemmatize(token) for token in listOfTokens]

# applies stemming to a list of tokenized words
def apply_stemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

# removes any words composed of less than 2 or more than 21 letters
def two_letters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 3 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord

is_noun = lambda pos: pos[:2] == 'NN'

@delayed
def process_one_entry_corpus(entry, language):
    '''function to procss text into list of token word'''
    words = set(nltk.corpus.words.words())
    
    stopwords = nltk.corpus.stopwords.words(language)
    param_lemmatizer = WordNetLemmatizer()
    param_stemmer = SnowballStemmer(language)
    
    other_words = [line.rstrip('\n') for line in open(f'{dir_path}/lists/stopwords_scrapmaker.txt')] # Load .txt file line by line

    entry = entry.replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
    entry = entry.replace(',', ' ')         # Removes commas
    entry = entry.rstrip('\n')              # Removes line breaks
    entry = entry.casefold()                # Makes all letters lowercase
    
    entry = re.sub(',|\'', ' ', entry)      # Removes commas and apostrophes
    entry = re.sub(r'\\n',' ', entry)       # Removes line breaks
    entry = re.sub(r'\\t',' ', entry)       # Removes tabs
    entry = re.sub('_+|=+','', entry)       # Removes underscore and equal signs
    entry = re.sub('\.+','', entry)         # Removes dots
    entry = re.sub(r'--+|\\|\/|\|','', entry)# Removes double dash, slash, backslash and pipe
    entry = re.sub('\W_',' ', entry)        # removes specials characters and leaves only words
    entry = re.sub("\S*\d\S*"," ", entry)   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
    entry = re.sub("\S*@\S*\s?"," ", entry) # removes emails and mentions (words with @)
    entry = re.sub(r'http\S+', '', entry)   # removes URLs with http
    entry = re.sub(r'www\S+', '', entry)    # removes URLs with www

    entry = " ".join(w for w,p in nltk.pos_tag(nltk.wordpunct_tokenize(entry)) \
                    if w.lower() in words or not w.isalpha() \
                    and is_noun(p))
    
    listOfTokens = word_tokenize(entry)
    twoLetterWord = two_letters(listOfTokens)

    listOfTokens = remove_words(listOfTokens, stopwords)
    listOfTokens = remove_words(listOfTokens, other_words)
    
    
    listOfTokens = apply_lemmatizer(listOfTokens, param_lemmatizer)
    # listOfTokens = apply_stemming(listOfTokens, param_stemmer)
    listOfTokens = remove_words(listOfTokens, twoLetterWord)
    listOfTokens = remove_words(listOfTokens, other_words)

    entry = " ".join(listOfTokens)
    entry = unidecode(entry)
    
    return entry

@delayed
def process_corpus(corpus, language: str) -> list:
    '''function that process bag corpus with loop of process 
        and return a list of delayed processes'''
    return [process_one_entry_corpus(entry, language) for entry in corpus]