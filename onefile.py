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

from scipy.sparse import csr_matrix
import numpy as np
# import pandas as pd # already imported
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count

class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        # to resolve SettingWithCopyWarning
        # dd = train.loc[train.user_id.isin(users)]
        # dd['ci'] = self.item_enc.transform(dd.item_id)
        # dd['cu'] = self.user_enc.transform(dd.user_id)
        dd = train.loc[train.user_id.isin(users)].copy()
        dd.loc[:, 'ci'] = self.item_enc.transform(dd.loc[:, 'item_id'])
        dd.loc[:, 'cu'] = self.user_enc.transform(dd.loc[:, 'user_id'])
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_id": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        return r

# import pandas as pd # already imported
# import numpy as np # already imported

def make_unary(df):
    '''
    Remove ratings below 3 and set ratings to 1.
    Following Anelli et al. "Top-N Recommendation Algorithms: A Quest for the State-of-the-Art"
    '''
    # Remove ratings below 3
    df = df[df['rating'] >= 3]
    # Set ratings to 1
    df.loc[:, 'rating'] = 1

    return df

def compute_p_core(df, p):
    """
    Compute the p-core of a dataset.
    
    Parameters:
    df (pandas.DataFrame): The input dataset with columns 'user_id' and 'item_id'.
    p (int): The minimum number of interactions for each user and item.
    
    Returns:
    pandas.DataFrame: The p-core of the dataset.
    """
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        df = df[df['user_id'].isin(user_counts[user_counts >= p].index)]
        df = df[df['item_id'].isin(item_counts[item_counts >= p].index)]

        if df['user_id'].value_counts().min() >= p and df['item_id'].value_counts().min() >= p:
            break

    return df

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import KFold

def objective(params):
    lambda_ = params['lambda_']
    folds = 5
    kfold = KFold(n_splits=folds, shuffle=True, random_state=7)
    ndcgs = []

    for train_index, test_index in kfold.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        # Fit the EASE model
        ease = EASE()
        ease.fit(train, lambda_=lambda_)

        # Generate recommendations
        items = df['item_id'].unique()
        recommendations = ease.predict(train, test['user_id'].unique(), items, k=10)
        print("Here's what recommendations look like:")
        print(recommendations.head())

        # Compute the nDCG for each user
        for user in test['user_id'].unique():
            recommended_items = recommendations.loc[recommendations['user_id'] == user, 'item_id']
            relevant_items = test.loc[test['user_id'] == user, 'item_id']
            # print("Here's relevant items:")
            # print(relevant_items)
            r = [1 if item in relevant_items else 0 for item in recommended_items]
            ndcgs.append(ndcg_at_k(r, 10))

    # Compute the average nDCG
    mean_ndcg = np.mean(ndcgs)

    return {'loss': -mean_ndcg, 'status': STATUS_OK}

if __name__ == "__main__":
    # Load the movielens-1m dataset
    print("Loading movielens-1m dataset...")
    df = get_movielens_1m()

    # make it unary
    print("Making the dataset unary...")
    df = make_unary(df)

    # Compute the 10-core of the dataset
    print("Computing 10-core...")
    df = compute_p_core(df, 10)

    # Define the search space for the hyperparameters
    space = {
        'lambda_': hp.loguniform('lambda_', np.log(2.72), np.log(16))
    }

    # Run the hyperparameter optimization
    print("Running hyperparameter optimization...")
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

    print(f'Best hyperparameters: {best}')