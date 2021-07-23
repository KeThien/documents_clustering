import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from wordcloud                        import WordCloud

from utils.parquet import read_parquet_to_df
k = 3
dir_path = os.path.dirname(os.path.realpath(__file__))
kmeans = pickle.load(open(f"{dir_path}/data/kmeans_{k}.pickle", "rb"))
df = read_parquet_to_df(f"{dir_path}/data/training_dataset_.parquet")

# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)        
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(f'{dir_path}/data/analysis/word_cloud_{i}')
        
centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids.columns = df.compute().columns
generateWordClouds(centroids)