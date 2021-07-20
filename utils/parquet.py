import os
import dask.dataframe as dd
from dask import delayed
from dask.distributed import progress
from utils.check_file import check_if_file_exists

dir_path = os.path.dirname(os.path.realpath(__file__))

@delayed
def openTsvXZ(file:str, columns:list):
    ''' function to open .tsv.xz file and return a DataFrame'''
    ddf = dd.read_csv(file, sep='\t', blocksize=None) # read_csv with Dask
    ddf.columns = columns
    ddf = ddf.repartition(npartitions=4) # partitions in chunk with Dask for better memory
    return ddf

def create_parquet_from_df(df, parquet_file_path:str) -> None:
    print(f'\nCreating {parquet_file_path} parquet file...')
    parquet = df.to_parquet(parquet_file_path, engine='pyarrow').compute()
    progress(parquet)
    
def read_parquet_to_df(file_path: str):
    '''Read parquet file to a DataFrame'''
    if check_if_file_exists(file_path):
        print(f'\nReading from {file_path} file...')
        df = dd.read_parquet(file_path, engine='pyarrow')
        return df
    else:
        raise Exception(f'{file_path} file does not exist')