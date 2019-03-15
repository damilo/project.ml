import os
import csv
import re
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer

def init_estimator ():
    print ('\r[i] init...', flush = True, end = ' ')
    # function that would load the estimator and prepare the vectorizer
    vectorizer = CountVectorizer (
        stop_words = 'english',
        max_features = 4096,
        binary = True
    )
    print ('done')

def predict (mailText):

    # preprocess mail text
    # ...
    #init_estimator ()

    print ('--------------------')
    print (mailText[:20], '[...]')

    # do something else
    # ...

    # return something
    ret = '{}, {}'.format (datetime.utcnow ().strftime ('%Y%m%d%H%M%S'), mailText[:20])
    return ret


if __name__ == '__main__':
    print (predict ("Hello Python"))