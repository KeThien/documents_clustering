import os
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

from utils.check_file import check_if_file_exists
from utils.parquet import read_parquet_to_df

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans
from dask_ml.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

dir_path = os.path.dirname(os.path.realpath(__file__))

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
        plt.savefig(f'{dir_path}/data/analysis/plot_most_common_words_in_cluster{i}.png')

def run_kmeans_then_pickle(df: dd.DataFrame, k: int = 5) -> KMeans:
    '''function to run kmeans save in pickle and return it
    :param df: dataframe to run kmeans on
    :param k: number of clusters
    return: KMeans object
    '''
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(df)
    pickle.dump(kmeans, open(f"{dir_path}/data/kmeans_{k}.pickle", "wb")) # Save model in pickle file
    return kmeans

if __name__ == '__main__':
    client = Client(n_workers=4,threads_per_worker=1, processes=False)
    
    df = read_parquet_to_df(f"{dir_path}/data/training_dataset_.parquet")
    
    print(f"\n{df.shape[0].compute()} rows")
    print(df.compute().T.nlargest(5, 0))
    
    k = 3
    run_kmeans_then_pickle(df, k)
    
    kmeans = pickle.load(open(f"{dir_path}/data/kmeans_{k}.pickle", "rb"))
    
    final_df_array = df.compute().to_numpy()
    prediction = kmeans.predict(df.compute())
    n_feats = 20
    
    dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
    plotWords(dfs, 13)
    
    client.close()