import os
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client, progress

from utils.open_pdf import openPdf2Text, openMultiPdf
from utils.process_corpus import process_corpus
from utils.tfidf import tfIdf_calculation
from utils.check_file import check_if_file_exists
from utils.parquet import openTsvXZ, create_parquet_from_df, read_parquet_to_df

dir_path = os.path.dirname(os.path.realpath(__file__))

list_files = [
    '0a1a1df06e4bd5d273647a880e2d3c4b.pdf', 
    '0a2d99f2d1af478ca00820eca51bc3a1.pdf', 
    '0a23d21e7d7c4d6bb177c1b5235a470f.pdf'
    ]


def process_tfidf_dataframe_to_parquet(parquet_file_name: str, n_take=False) -> None:
    '''Create a Dataframe from in.tsv.xz
    process to stemming / lemmatizer and TF-IDF then save it to parquet file'''
    
    if not check_if_file_exists(f'{dir_path}/data/{parquet_file_name}_{n_take}.parquet'):
        columns_for_df = ['filename', 'keys', 'text_djvu', 'text_tesseract', 'text_textract', 'text_best']
        df = openTsvXZ(f'{dir_path}/data/in.tsv.xz', columns_for_df)
        corpus = df['text_best'].to_bag().take(n_take) if n_take else df['text_best'].to_bag()
        print('--------------Creating dataframe and tf-idf it------------------')
        corpus = process_corpus(corpus, 'english')
        result = corpus.persist()
        progress(result)
        result = [x.compute() for x in result.compute()]
        df = tfIdf_calculation(result)
        progress(create_parquet_from_df(df.persist(), parquet_file_name))
    else:
        print(f'\n{parquet_file_name} already exists')

if __name__ == '__main__':
    client = Client(n_workers=4,threads_per_worker=1, processes=False)  
    prefix_file_name = 'training_dataset'
    n_take = 5 # number of files to read, 0 = all
    parquet_file_name = f"{dir_path}/data/{prefix_file_name}_{n_take if n_take > 0 else ''}.parquet"
    
    if not check_if_file_exists(parquet_file_name):
        process_tfidf_dataframe_to_parquet(parquet_file_name, n_take)
    else:
        df = read_parquet_to_df(parquet_file_name)
    
        print(f"\n{df.shape[0].compute()} rows")
        # print(df.compute().T.nlargest(5, 0))
    
    client.close()