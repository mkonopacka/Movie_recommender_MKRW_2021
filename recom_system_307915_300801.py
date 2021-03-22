# %%
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.decomposition import NMF, TruncatedSVD
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

def RMSE(Zp):
    ''' Root-mean-square error between matrix Zp and test matrix '''
    RMSE = 0
    for row in range(test.shape[0]):
        i = test.userId[row] - 1
        j = all_movies.index(test.movieId[row])
        k = test.rating[row]
        RMSE = RMSE + (Zp[i][j] - k)**2
    return sqrt(RMSE/test.shape[0])

# %% Algorytmy
def test_NMF(Z_, r = 10, print = False):
    ''' Nonnegative Matrix Factorization 
        Z_(nd.array) original matrix
        r (float) number of features'''
    if print: print('Building a model...')
    model = NMF(n_components = r, init = 'random', random_state = 77)
    W = model.fit_transform(Z_)
    H = model.components_
    Z_approx = np.dot(W,H)
    if print: print('Model finished.')
    result = RMSE(Z_approx)
    if print: print(f"RMSE for matrix Z' (Z approximated with NMF): {result}")
    return result

def test_SVD1(Z_, r = 3, print = False):
    ''' Singular Value Decomposition; r: rank '''
    if print: print('Building a model...')
    U, S, VT = np.linalg.svd(Z_, full_matrices = False)
    S = np.diag(S)
    Z_approx = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    if print: print('Model finished.')
    result = RMSE(Z_approx)
    if print: print(f"RMSE for matrix Z' (Z approximated with SVD1 with rank = {r}): {result}")
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

# %% Create matrix Z_avg: wypełnianie średnim rankingiem spośród wszystkich ocenionych filmów w zbiorze treningowym
avg_rating = np.mean(train.rating) 
Z_avg = np.full((n,d), avg_rating)
for row in tqdm(range(train.shape[0])):
    ''' Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
        jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście '''
    i = train.userId[row] - 1
    j = all_movies.index(train.movieId[row])
    k = train.rating[row]
    Z_avg[i][j] = k

# print('RMSE for original matrix Z_avg: ', RMSE(Z_avg))
# %% Create matrix Z_avg_user: wypełnianie średnią oceną dla danego użytkownika w zbiorze treningowym
user_avgs = np.array(train.groupby('userId')['rating'].mean())
Z_avg_user = np.repeat(user_avgs, d).reshape(n,d)
for row in tqdm(range(train.shape[0])):
    ''' Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
        jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście '''
    i = train.userId[row] - 1
    j = all_movies.index(train.movieId[row])
    k = train.rating[row]
    Z_avg_user[i][j] = k

# print('RMSE for original matrix Z_avg_user: ', RMSE(Z_avg_user))

# # %% TODO Create matrix Z_avg_movie: wypełnianie średnią oceną dla danego filmu w zbiorze treningowym
# #         i średnią wszystkich filmów dla filmów których w nim nie ma
movie_avgs = train.groupby('movieId')['rating'].mean()
movie_row = np.repeat(avg_rating, d)
for id, rating in movie_avgs.iteritems():
    movie_row[all_movies.index(id)] = rating
Z_avg_movie = np.array([movie_row]*n)

# # missing_ids = [id for id in test_movies if not id in train_movies]
# # missing_avgs = pd.Series(np.repeat(avg_rating, len(missing_ids)), index= missing_ids)
# movie_avgs = np.array(movie_avgs.append(missing_avgs)) # array ze średnimi wszystkich filmów 9724
# # Tworzymy macierz przez powtórzenie wiersza średnich ocen filmów 610 razy
# # a następnie wypełnienie znanymi wartościami ze zbioru treningowego
# Z_avg_movie = np.array([movie_avgs]*n)
for row in tqdm(range(train.shape[0])):
    ''' Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
        jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście '''
    i = train.userId[row] - 1
    j = all_movies.index(train.movieId[row])
    k = train.rating[row]
    Z_avg_user[i][j] = k

# print('RMSE for original matrix Z_avg_movie: ', RMSE(Z_avg_movie))
# %% TODO Create matrix Z_perc: wypełnianie oceną odpowiadającą percentylem oceny filmu ocenie użytkownika

# %% TODO średnia po filmie i po użytkowniku 2 sposoby
# %% TESTY
for Z in [Z_avg, Z_avg_movie, Z_avg_user]:
    print(test_SVD1(Z))
    print(test_NMF(Z))
# %% Program
'''
if args.alg == 'NMF': 
    test_NMF(Z_avg, print = True)

if args.alg == 'SVD1': 
    test_SVD1(Z_avg, 3, print = True)
'''