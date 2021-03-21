# %% Load data, import libs, define functions
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.decomposition import NMF
from math import sqrt

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'MKRW 2021 Projekt 1')
    parser.add_argument('--train', help = 'train file', default = 'data/ratings_train.csv')
    parser.add_argument('--test', help = 'test file', default = 'data/ratings_test.csv')
    parser.add_argument('--alg', help = '\'NMF (default)\' or \'SVD1\' or \'SVD2\' or \'SGD\'', default = 'NMF')
    parser.add_argument('--result_file', help = 'file where final RMSE will be saved', default = 'result.txt')
    args, unknown = parser.parse_known_args() # makes it possible to use inside interactive Python kernel
    print(f'Arguments unrecognized: {unknown}.\nReturning recognized arguments: {args}.') 
    return args

args = parse_arguments()
train = pd.read_csv(args.train, index_col= False)
test = pd.read_csv(args.test, index_col= False)

# %% Create training matrix n x d (zeros represent NA)
train_movies = train.movieId.unique()
test_movies = test.movieId.unique()
all_movies = sorted(np.concatenate((train_movies, test_movies[~ np.isin(test_movies, train_movies)])))
d = len(all_movies) # 9724
train_users = train.userId.unique()
test_users = test.userId.unique()
all_users = np.concatenate((train_users, test_users[~ np.isin(test_users, train_users)]))
n = len(all_users) # 610

avg_rating = np.mean(train.rating)
Z = np.full((n,d), avg_rating)
for row in tqdm(range(train.shape[0])):
    ''' Uwaga: Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
    jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście) '''
    i = train.userId[row] - 1
    j = all_movies.index(train.movieId[row])
    k = train.rating[row]
    Z[i][j] = k
    
    
RMSE = 0
for row in tqdm(range(test.shape[0])):
    i = test.userId[row] - 1
    j = all_movies.index(test.movieId[row])
    k = test.rating[row]
    RMSE = RMSE + (Z[i][j] - k)**2
    
RMSE = sqrt(RMSE/test.shape[0])
print('RMSE for original matrix Z: ', RMSE)

if args.alg=='NMF':
    print('Building a model...')
    model = NMF(n_components=10, init='random', random_state=77)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W,H)
    print('Model finished.')

RMSE = 0
for row in tqdm(range(test.shape[0])):
    i = test.userId[row] - 1
    j = all_movies.index(test.movieId[row])
    k = test.rating[row]
    RMSE = RMSE + (Z_approx[i][j] - k)**2
    
RMSE = sqrt(RMSE/test.shape[0])
print("RMSE for matrix Z' (Z approximated with NMF): ", RMSE)
