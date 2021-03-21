# %%
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
    return args

args = parse_arguments()
train = pd.read_csv(args.train, index_col= False)
test = pd.read_csv(args.test, index_col= False)

# %% Algorytmy
def RMSE(Zp):
    ''' Root-mean-square error between matrix Zp and test matrix '''
    RMSE = 0
    for row in range(test.shape[0]):
        i = test.userId[row] - 1
        j = all_movies.index(test.movieId[row])
        k = test.rating[row]
        RMSE = RMSE + (Zp[i][j] - k)**2
    return sqrt(RMSE/test.shape[0])

def test_NMF(Z_, r = 10, comm = False):
    ''' Nonnegative Matrix Factorization 
        Z_(nd.array) original matrix
        r (float) number of features'''
    if comm: print('Building a model...')
    model = NMF(n_components = r, init = 'random', random_state = 77)
    W = model.fit_transform(Z_)
    H = model.components_
    Z_approx = np.dot(W,H)
    if comm: print('Model finished.')
    result = RMSE(Z_approx)
    if comm: print(f"RMSE for matrix Z' (Z approximated with NMF): {result}")
    return result

def SVD(Z, r):
    U, S, VT = np.linalg.svd(Z, full_matrices = False)
    S = np.diag(S)
    Z_approx = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    return Z_approx

def test_SVD1(Z_, r = 3, comm = False):
    ''' Singular Value Decomposition; r: rank '''
    if comm: print('Building a model...')
    Z_approx = SVD(Z_, r)
    if comm: print('Model finished.')
    result = RMSE(Z_approx)
    if comm: print(f"RMSE for matrix Z' (Z approximated with SVD1 with rank = {r}): {result}")
    return result
    
def test_SVD2(Z_, i=3, r=5, comm = False):
    ''' SVD2; i: number of iterations; r: rank'''
    if comm: print('Building a model...')
    Z_approx = np.copy(Z_)
    for j in tqdm(range(i)):
        Z_approx = SVD(Z_approx, r)
        if j != i-1: train_matrix(Z_approx)
    if comm: print('Model finished.')
    result = RMSE(Z_approx)
    if comm: print(f"RMSE for matrix Z' (Z approximated with SVD2 with rank = {r}, nr of iterations = {i}): {result}")
    return result

# %% 
train_movies = train.movieId.unique()
test_movies = test.movieId.unique()
all_movies = sorted(np.concatenate((train_movies, test_movies[~ np.isin(test_movies, train_movies)])))
d = len(all_movies) # 9724
train_users = train.userId.unique()
test_users = test.userId.unique()
all_users = np.concatenate((train_users, test_users[~ np.isin(test_users, train_users)]))
n = len(all_users) # 610

# %% Create matrix Z_avg
avg_rating = np.mean(train.rating) 
Z_avg = np.full((n,d), avg_rating)

def train_matrix(Z):
    '''fills matrix Z with training data'''
    for row in tqdm(range(train.shape[0])):
        ''' Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
            jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście '''
        i = train.userId[row] - 1
        j = all_movies.index(train.movieId[row])
        k = train.rating[row]
        Z[i][j] = k
    return 

#train_matrix(Z_avg)
#print('RMSE for original matrix Z: ', RMSE(Z_avg))

# %% Create matrix Z_avg_user: wypełnianie średnią oceną dla danego użytkownika w zbiorze treningowym
user_avgs = np.array(train.groupby('userId')['rating'].mean())
Z_avg_user = np.repeat(user_avgs, d).reshape(n,d)
train_matrix(Z_avg_user)
print('RMSE for original matrix Z_avg_user: ', RMSE(Z_avg_user))

# %% TODO Create matrix Z_avg_user: wypełnianie średnią oceną dla danego użytkownika

# %% TODO Create matrix Z_avg_movie: wypełnianie średnią oceną dla danego filmu

# %% TODO Create matrix Z_perc: wypełnianie oceną odpowiadającą percentylem oceny filmu ocenie użytkownika

# %% TODO Create matrix Z_cluster: wypełnianie średnią oceną w jakiejś grupie filmów? Do tego potrzebny clustering
# %% TESTY

# %% Program

if args.alg == 'NMF': 
    test_NMF(Z_avg_user, r=10, comm = True)

if args.alg == 'SVD1': 
    test_SVD1(Z_avg_user, r=10, comm = True)

if args.alg == 'SVD2': 
    test_SVD2(Z_avg_user, i=3, r=10, comm = True)