import os
from typing import Mapping
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask import compute, delayed
from dask.distributed import Client, progress
import dask.bag as db
import dask.array as da

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from utils.open_pdf import openPdf2Text, openMultiPdf
from utils.process_corpus import process_corpus
from utils.tfidf import tfIdf_calculation

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans
from dask_ml.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

dir_path = os.path.dirname(os.path.realpath(__file__))

list_files = [
    '0a1a1df06e4bd5d273647a880e2d3c4b.pdf', 
    '0a2d99f2d1af478ca00820eca51bc3a1.pdf', 
    '0a23d21e7d7c4d6bb177c1b5235a470f.pdf'
    ]
def check_if_file_exists(file_name:str) -> bool:
    return os.path.isfile(f'{dir_path}/data/{file_name}')

@delayed
def openTsvXZ(file:str, columns:list):
    ''' function to open .tsv.xz file and return a DataFrame'''
    ddf = dd.read_csv(file, sep='\t', blocksize=None) # read_csv with Dask
    ddf.columns = columns
    ddf = ddf.repartition(npartitions=4) # partitions in chunk with Dask for better memory
    return ddf

def create_parquet_from_df(df, parquet_file_name:str) -> None:
    print(f'\nCreating {parquet_file_name} parquet file...')
    parquet = df.to_parquet(f'{dir_path}/data/{parquet_file_name}.parquet', engine='pyarrow').compute()
    progress(parquet)

def read_parquet_to_df(parquet_file_name: str):
    '''Read parquet file to a DataFrame'''
    if check_if_file_exists(f'{parquet_file_name}.parquet'):
        print(f'Reading from {parquet_file_name} file...')
        df = dd.read_parquet(f'{dir_path}/data/{parquet_file_name}.parquet', engine='pyarrow')
        return df
    else:
        print(f'{parquet_file_name} does not exist')

def process_tfidf_dataframe_to_parquet(parquet_file_name: str, n_take=False) -> None:
    '''Create a Dataframe from in.tsv.xz
    process to stemming and TF-IDF then save it to parquet file'''
    
    if not check_if_file_exists(f'{parquet_file_name}_{n_take}.parquet'):
        columns_for_df = ['filename', 'keys', 'text_djvu', 'text_tesseract', 'text_textract', 'text_best']
        df = openTsvXZ(f'{dir_path}/data/in.tsv.xz', columns_for_df)
        corpus = df['text_best'].to_bag().take(n_take) if n_take else df['text_best'].to_bag()
        print('--------------Creating dataframe and tf-idf it------------------')
        corpus = process_corpus(corpus, 'english')
        result = corpus.persist()
        progress(result)
        result = [x.compute() for x in result.compute()]
        print(result)
        df = tfIdf_calculation(result)
        # progress(df.persist())
        progress(create_parquet_from_df(df.persist(), parquet_file_name))
    else:
        print(f'\n{parquet_file_name} already exists')

def get_top_features_cluster(tf_idf_array, prediction, n_feats:int):
    print(prediction)
    labels = np.unique(prediction.compute())
    dfs = []
    vectorizer = pickle.load(open(f"{dir_path}/data/vectorizer.pickle", 'rb'))
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats:int) -> None:
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.savefig(f'plot_most_common_words_in_cluster{i}.png')

if __name__ == '__main__':
    client = Client(n_workers=4,threads_per_worker=1, processes=False)  
    parquet_file_name = 'my_parquet_10'
    n_take = 10
    
    if not check_if_file_exists(f'{parquet_file_name}.parquet'):
        process_tfidf_dataframe_to_parquet(parquet_file_name, n_take if n_take > 0 else '')
        
    df = read_parquet_to_df(parquet_file_name)
    
    print(f"\n{df.shape[0].compute()} rows")
    print(df.compute().T.nlargest(5, 0))
    
    
    # k = 5
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(df.compute())
    
    # final_df_array = df.compute().to_numpy()
    # prediction = kmeans.predict(df.compute())
    # n_feats = 20
    
    # dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
    # plotWords(dfs, 13)
    
    client.close()