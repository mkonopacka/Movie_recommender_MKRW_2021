# %% Load data, import libs, define functions
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'MKRW 2021 Projekt 1')
    parser.add_argument('--train', help = 'train file', default = 'data/ratings_train.csv')
    parser.add_argument('--test', help = 'test file', default = 'data/ratings_test.csv')
    parser.add_argument('--alg', help = '\'NMF (default)\' or \'SVD1\' or \'SVD2\' or \'SGD\'', default = 'NMF')
    parser.add_argument('--result_file', help = 'file where final RMSE will be saved', default = 'result.txt')
    args, unknown = parser.parse_known_args() # makes it possible to use inside interactive Python kernel
    print(f'Arguments unrecognized: {unknown}.\nReturning recognized arguments: {args}.') 
    return args

def RMSE(Z, T):
    ''' Calculate root-mean squared error between two numpy arrays of the same shape'''
    if not (type(Z) == 'nd.array' and type(Z) == type(T)): raise TypeError('Z and T must be numpy arrays') 
    if not Z.shape == T.shape: raise ValueError('Z.shape and T.shape must be the same')
    
    T_pairs = T.shape[0]*T.shape[1]
    return np.sqrt(sum((Z - T)^2 )/T_pairs)

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

train_matrix = np.zeros(shape = (n, d))
for row in tqdm(range(train.shape[0])):
    ''' Uwaga: Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
    jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście) '''
    i = train.userId[row] - 1
    j = all_movies.index(train.movieId[row])
    k = train.rating[row]
    train_matrix[i][j] = k
    