import pandas as pd
import numpy as np

def make_unary(df):
    '''
    Remove ratings below 3 and set ratings to 1.
    Following Anelli et al. "Top-N Recommendation Algorithms: A Quest for the State-of-the-Art"
    '''
    # Remove ratings below 3
    df = df[df['rating'] >= 3]
    # Set ratings to 1
    df['rating'] = 1

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

if __name__ == "__main__":
    from model import EASE
    from get_movielens_1m import get_movielens_1m
    # Load the movielens-1m dataset
    print("Loading movielens-1m dataset...")
    df = get_movielens_1m()

    # make it unary
    print("Making the dataset unary...")
    df = make_unary(df)

    # Compute the 10-core of the dataset
    print("Computing 10-core...")
    df = compute_p_core(df, 10)

    # Split the data into a training set and a test set
    print("Splitting the data into a training set and a test set...")
    train = df.sample(frac=0.8, random_state=7) 
    test = df.drop(train.index)

    # Fit the EASE model
    ease = EASE()
    print("Fitting the EASE model...")
    ease.fit(train)

    # Generate recommendations
    print("Generating recommendations...")
    items = df['item_id'].unique()
    recommendations = ease.predict(train, test['user_id'].unique(), items, k=10)

    # Compute the nDCG for each user
    print("Computing the nDCG for each user...")
    ndcgs = []
    for user in test['user_id'].unique():
        recommended_items = recommendations.loc[recommendations['user_id'] == user, 'item_id']
        relevant_items = test.loc[test['user_id'] == user, 'item_id']
        r = [1 if item in relevant_items else 0 for item in recommended_items]
        ndcgs.append(ndcg_at_k(r, 10))

    # Compute the average nDCG
    print("Computing the average nDCG...")
    mean_ndcg = np.mean(ndcgs)
    print(f'Mean nDCG: {mean_ndcg}')