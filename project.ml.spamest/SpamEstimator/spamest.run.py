import os
import csv
import re
from mlpspafi import MlpSpafi
from sklearn.feature_extraction.text import CountVectorizer


def get_vectorizer ():
    FILE_DS = os.path.join ('src', 'emails.csv')

    header = True
    X_raw = []
    y_raw = []
    with open (FILE_DS) as csvfile:
        reader = csv.reader (csvfile, delimiter=',')
        for row in reader:
            
            # jump over header
            if (header):
                header = False
                continue
            
            # col "text"
            X_raw.append (row[0])
    
    # remove string 'Subject: '
    for i, text in enumerate (X_raw):
        X_raw[i] = re.sub (r'Subject: ', '', text)

    vectorizer = CountVectorizer (
        stop_words = 'english',
        max_features = 4096,
        binary = True
    ).fit (X_raw)

    return vectorizer


if __name__ == '__main__':
    
    mlp_best = MlpSpafi (name = 'MlpSpafi', restore = True)
    vectorizer = get_vectorizer ()

    fileName = os.path.join ('..', '_mails', 'mailBody.txt')
    lastModified = os.path.getmtime (fileName)
    print ('[i] ready...')
    while True:
        # run until Ctrl+C
        
        thisModified = os.path.getmtime (fileName)
        if (thisModified != lastModified):
            with open (fileName, 'r') as mailBody:
                print ('--------------------')
                mailText = mailBody.read ()
                print (mailText[:200], '[...]')
                X_pred = vectorizer.transform ([mailText]).toarray ().astype ('f4')
                print (X_pred.shape, X_pred, X_pred.sum ())

                y_pred, y_pred_distr = mlp_best.predict (X_pred)
                print ('[>] result:', '{}'.format ('spam' if y_pred else 'ham'), y_pred_distr)
            
            lastModified = thisModified