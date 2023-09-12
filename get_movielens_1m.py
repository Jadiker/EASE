import pandas as pd
import urllib.request
import zipfile
import os

def get_movielens_1m():
    '''
    Get the movielens-1m dataset.
    '''
    # Check if the dataset is already downloaded
    if not os.path.exists('ml-1m/ratings.dat'):
        # Download the dataset
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        filename = 'ml-1m.zip'
        urllib.request.urlretrieve(url, filename)

        # Unzip the dataset
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()

    # Load the dataset
    df = pd.read_csv('ml-1m/ratings.dat', delimiter='::', header=None, 
                     names=['user_id', 'item_id', 'rating', 'timestamp'], 
                     engine='python')

    # Drop the timestamp column as we don't need it
    df.drop('timestamp', axis=1, inplace=True)

    return df

if __name__ == '__main__':
    df = get_movielens_1m()
    print(df.head())