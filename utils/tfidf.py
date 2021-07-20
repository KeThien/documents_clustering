import os
import pandas as pd
from dask import delayed
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

dir_path = os.path.dirname(os.path.realpath(__file__))

@delayed
def tfIdf_calculation(corpus):
    ''' function to calculate TF-IDF from list of strings and return a DataFrame'''
    print('TF-IDF calculation...')
    vectorizer=TfidfVectorizer(
        use_idf = True, 
        smooth_idf = True,
        stop_words = [
            'the', 'we', 'to', 'and', 'of', 'they', 'was', 'at', 'for', 'in', 'p3', 'have'
        ]
        )
    with joblib.parallel_backend('dask'):
        X = vectorizer.fit_transform(corpus)
        pickle.dump(vectorizer, open(f"{dir_path}/../data/vectorizer.pickle", "wb")) #Save vectorizer
        tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names())
    final_df = tf_idf
    return final_df