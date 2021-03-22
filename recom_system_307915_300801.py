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
train = pd.read_csv(args.train, index_col = False)
test = pd.read_csv(args.test, index_col = False)

def RMSE(Zp):
    ''' Root-mean-square error between matrix Zp and test matrix '''
    RMSE = 0
    for row in range(test.shape[0]):
        i = test.userId[row] - 1
        j = all_movies.index(test.movieId[row])
        k = test.rating[row]
        RMSE = RMSE + (Zp[i][j] - k)**2
    return sqrt(RMSE/test.shape[0])

def fill_matrix(Z):
    ''' Fills matrix Z with known entries from training data '''
    for row in tqdm(range(train.shape[0])):
        ''' Id userów są od 1 do 610 i jest ich 610 więc odejmujemy 1, a Id filmów
            jest dużo mniej niż ich max. więc zamiast id bierzemy jego numer w posortowanej liście '''
        i = train.userId[row] - 1
        j = all_movies.index(train.movieId[row])
        k = train.rating[row]
        Z[i][j] = k
    return

# %% Algorytmy (wszystkie approx_ zwracaja macierz, a test_ wynik RMSE)
def approx_NMF(Z_, r = 10):
    ''' Nonnegative Matrix Factorization; Return approximated matrix;
        Z_(nd.array) original matrix
        r (float) number of features '''
    model = NMF(n_components = r, init = 'random', random_state = 77)
    W = model.fit_transform(Z_)
    H = model.components_
    return np.dot(W,H)

def test_NMF(Z_, r = 10, log = False):
    if log: print('Building a model...')
    Z_approx = approx_NMF(Z_, r)
    if log: print('Model finished.')
    result = RMSE(Z_approx)
    if log: print(f"RMSE for matrix Z' (Z approximated with NMF): {result}")
    return result

def approx_SVD1(Z_, r = 3):
    ''' Singular Value Decomposition; Return approximated matrix;
        Z_(nd.array) original matrix
        r (float) number of features'''
    U, S, VT = np.linalg.svd(Z_, full_matrices = False)
    S = np.diag(S)
    return U[:,:r] @ S[0:r,:r] @ VT[:r,:]

def test_SVD1(Z_, r = 3, log = False):
    if log: log('Building a model...')
    Z_approx = approx_SVD1(Z_, r)
    if log: log('Model finished.')
    result = RMSE(Z_approx)
    if log: log(f"RMSE for matrix Z' (Z approximated with SVD1 with rank = {r}): {result}")
    return result

def approx_SVD2(Z_, i = 3, r = 5):
    ''' SVD with iterations; Return approximated matrix
        Z_(nd.array) original matrix
        r (float) number of features
        i (int) number of iterations'''
    Z_approx = np.copy(Z_)
    for j in tqdm(range(i)):
        Z_approx = approx_SVD1(Z_approx, r)
        if j != i-1: 
            fill_matrix(Z_approx)
    return Z_approx

def test_SVD2(Z_, i = 3, r = 5, log = False):
    if log: print('Building a model...')
    Z_approx = approx_SVD2(Z_, i, r)
    if log: print('Model finished.')
    result = RMSE(Z_approx)
    if log: print(f"RMSE for matrix Z' (Z approximated with SVD2 with rank = {r}, nr of iterations = {i}): {result}")
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
fill_matrix(Z_avg)
print('RMSE for original matrix Z_avg: ', RMSE(Z_avg))

# %% Create matrix Z_avg_user: wypełnianie średnią oceną dla danego użytkownika w zbiorze treningowym
user_avgs = np.array(train.groupby('userId')['rating'].mean())
Z_avg_user = np.repeat(user_avgs, d).reshape(n,d)
fill_matrix(Z_avg_user)
print('RMSE for original matrix Z_avg_user: ', RMSE(Z_avg_user))

# %% Create matrix Z_avg_movie: wypełnianie średnią oceną dla danego filmu w zbiorze treningowym
#    i średnią wszystkich filmów dla filmów których w nim nie ma
movie_avgs = train.groupby('movieId')['rating'].mean()
movie_row = np.repeat(avg_rating, d)
for id, rating in movie_avgs.iteritems(): movie_row[all_movies.index(id)] = rating
Z_avg_movie = np.array([movie_row]*n)
fill_matrix(Z_avg_movie)
print('RMSE for original matrix Z_avg_movie: ', RMSE(Z_avg_movie))

# %% TODO Create matrix Z_perc: wypełnianie oceną odpowiadającą percentylem oceny filmu ocenie użytkownika

# %% TODO Create matrix Z_avg2 średnia po filmie i po użytkowniku 2 sposoby

# %% Tests
# Zs = {'Z_avg': Z_avg,'Z_avg_movie': Z_avg_movie,'Z_avg_user': Z_avg_user}
# for name, Z in Zs.items():
#     print(f'SVD1 - {name} - {test_SVD1(Z)}')
#     print(f'NMF  - {name} - {test_NMF(Z)}')
#     print(f'SVD2 - {name} - {test_SVD2(Z)}')

# %% Program
# Wykomentowane na czas testów
# if args.alg == 'NMF': 
#     test_NMF(Z_avg_user, log = T, return_matrix = Truerue)

# if args.alg == 'SVD1': 
#     test_SVD1(Z_avg_user, r = 3, log = True)

# if args.alg == 'SVD2': 
#     test_SVD2(Z_avg_user, i = 3, r = 10, log = True)
