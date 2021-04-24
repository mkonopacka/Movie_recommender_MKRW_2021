# %% Import
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.decomposition import NMF, TruncatedSVD

# %% Parse arguments
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

# %% Functions using ratings.csv data
def RMSE(Zp):
    ''' Root-mean-square error between matrix Zp and test matrix '''
    RMSE = 0
    for row in range(test.shape[0]):
        i = test.userId[row] - 1
        j = all_movies.index(test.movieId[row])
        k = test.rating[row]
        RMSE = RMSE + (Zp[i][j] - k)**2
    return np.sqrt(RMSE/test.shape[0])

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
    #return W, H

def approx_SVD1(Z_, r = 3):
    ''' Singular Value Decomposition; Return approximated matrix;
        Z_(nd.array) original matrix
        r (float) number of features'''
    U, S, VT = np.linalg.svd(Z_, full_matrices = False)
    S = np.diag(S)
    #return U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    return U[:,:r], S[0:r,:r] @ VT[:r,:]
    
def approx_SVD1_b(Z_, r = 3):
    ''' Singular Value Decomposition; Return approximated matrix;
        Z_(nd.array) original matrix
        r (float) number of features
        uses sklearn'''
    svd = TruncatedSVD(n_components = r, random_state=42)
    svd.fit(Z_)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z_) / svd.singular_values_
    H = np.dot(Sigma2, VT) 
    return np.dot(W, H)

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
    
def test_alg(Z_, alg, r = 5, i = 3, log = False):
    if log: print('Building a model...')
    if alg == 'NMF': Z_approx = approx_NMF(Z_, r)
    if alg == 'SVD1': Z_approx = approx_SVD1(Z_, r)
    if alg == 'SVD2': Z_approx = approx_SVD2(Z_, i, r)
    if log: print('Model finished.')
    result = RMSE(Z_approx)
    if log: print(f"RMSE for matrix Z' (Z approximated with {alg} with rank = {r}): {result}")
    if alg == 'SVD2': print('Number of iterations: ', i)
    return result

# %% Setup
train_movies = train.movieId.unique()
test_movies = test.movieId.unique()
all_movies = sorted(np.concatenate((train_movies, test_movies[~ np.isin(test_movies, train_movies)])))
d = len(all_movies) # 9724
train_users = train.userId.unique()
test_users = test.userId.unique()
all_users = np.concatenate((train_users, test_users[~ np.isin(test_users, train_users)]))
n = len(all_users) # 610
avg_rating = np.mean(train.rating)

# %% Create matrix Z_avg: wypełnianie średnim rankingiem spośród wszystkich ocenionych filmów w zbiorze treningowym
avg_rating = np.mean(train.rating) 
Z_avg = np.full((n,d), avg_rating)
fill_matrix(Z_avg)
print('RMSE for original matrix Z_avg: ', RMSE(Z_avg))

# %% Create matrix Z_avg_user: wypełnianie średnią oceną dla danego użytkownika w zbiorze treningowym
user_avgs = np.array(train.groupby('userId')['rating'].mean())
Z_avg_user = np.repeat(user_avgs, d).reshape(n,d)
fill_matrix(Z_avg_user)
print('RMSE for original matrix Z_avg_user: ', RMSE(Z_avg_user))

# %% Create matrix Z_avg_movie: wypełnianie średnią oceną dla danego filmu w zbiorze treningowym
#    i średnią wszystkich filmów dla filmów których w nim nie ma
movie_avgs = train.groupby('movieId')['rating'].mean()
movie_row = np.repeat(avg_rating, d)
for id, rating in movie_avgs.iteritems(): movie_row[all_movies.index(id)] = rating
Z_avg_movie = np.array([movie_row]*n)
fill_matrix(Z_avg_movie)
print('RMSE for original matrix Z_avg_movie: ', RMSE(Z_avg_movie))

# %% Weighted mean of Z_avg_user and Z_avg_movie.
p = 0.58    # optimal parameter
Z_avg_user_movie = p*Z_avg_user + (1-p)*Z_avg_movie
print('RMSE for original matrix Z_avg_user_movie: ', RMSE(Z_avg_user_movie))


# %% Fills matrix with mean ratings of the most similar users.
def close_users(Z_, percent = 0.15):
    Z_close_users = np.full((n,d), 0)
    for i in tqdm(range(n)):
        user = Z_[i,]
        Z_user = (Z_ - user)**2
        distances = Z_user.sum(axis=1)
        q = np.quantile(distances, percent)
        indexes = distances <= q
        enum = np.arange(len(distances))
        indexes = enum[indexes]
        close_users = Z_[indexes]
        prediction = close_users.mean(axis=0)
        Z_close_users[i,] = prediction
    return Z_close_users

Z_close_users = close_users(Z_avg_user)
print('RMSE for original matrix Z_close_users: ', RMSE(Z_close_users))


# %% Enhanced Z_avg_user (not better)
user_avgs = np.array(train.groupby('userId')['rating'].mean())
Z_avg_user = np.repeat(user_avgs, d).reshape(n,d)
Z_avg_user_0 = np.copy(Z_avg_user)
#fill_matrix(Z_avg_user)
Z_avg_user_2 = Z_avg_movie - Z_avg_user_0
movie_corection = Z_avg_user_2.mean(axis=0)
Z_avg_user_2 = Z_avg_user_0 + movie_corection
print('RMSE for original matrix Z_avg_user_2: ', RMSE(Z_avg_user_2))

Z_close_users = close_users(Z_avg_user_2)
print('RMSE for original matrix Z_close_users: ', RMSE(Z_close_users))


# %% SGD - setup
Z_zero = np.full((n,d), 0, dtype = np.float)
fill_matrix(Z_zero)
entries = Z_zero > 0
vec_zero = Z_zero[entries]
pairs = list(zip(*np.where(entries)))

# %% SGD

# 1. updatowanie funkcji loss wzgledem kolumny/wiersza
# 2. nie wszystkie wspolzedne w jednym batchu
# 3. pochodna po kilku wspl. na raz
# 4. licznie pochodnej jednostronnej + wielokrotnie uzycie f(x) z pamieci

def vec_to_mat0(x):
    #r = len(x)//(n+d)
    W = x[0 : n*r]
    W = W.reshape((n, r))
    H = x[n*r : r*(n+d)]
    H = H.reshape((r, d))
    return W, H

def vec_to_mat(x):
    #r = len(x)//(n+d)
    W = x[0 : n*r]
    W = W.reshape((n, r))
    H = x[n*r : r*(n+d)]
    H = H.reshape((r, d))
    return W @ H

# how not to calculate whole W x H ?
def loss(x):
    WH = vec_to_mat(x)
    vec_wh = WH[entries]
    vec = (vec_zero - vec_wh)**2
    return np.sum(vec)

def loss0(x):
    W, H = vec_to_mat0(x)
    s = 0
    for i, j in pairs:
        s += (Z_zero[i,j] - np.sum(W[i,:] * H[:,j]))**2  
    return s

def loss_batch(x, ks, h):
    pass
    
def loss_part(W, H, k, h):
    s = 0
    if k < n*r:
        i0, j0 = k//r, (k+1) % r - 1
        row = np.copy(W[i0,:])
        row[j0] += h
        for i, j in pairs:
            if i==i0: s += (Z_zero[i,j] - np.sum(row * H[:,j]))**2
    else:
        k = k - n*r
        i0, j0 = k//d, (k+1) % d - 1
        col = np.copy(H[:,j0])
        col[i0] += h
        for i, j in pairs:
            if j==j0: s += (Z_zero[i,j] - np.sum(W[i,:] * col))**2 
    return s

r = 5
#x0 = np.full(r*(n+d), np.sqrt(avg_rating))
x0 = np.full(r*(n+d), 0.825)

#r = 10
#x0 = np.full(r*(n+d), 0.6) 

# could be twice as fast by using one-sided difference
def der_loss(x, ks, h):
    xp, xm = np.copy(x), np.copy(x)
    xp[ks] += h
    xm[ks] -= h
    return (loss(xp) - loss(xm))/(2*h)
            
# batch_size should divide N=r*(n+d)
def SGD0(x0, batch_size=1, l_rate=0.01, h=0.01, n_epochs=50):
    N = len(x0)
    n_iter = N//batch_size
    indexes = np.arange(N)
    x = np.copy(x0)
    for _ in tqdm(range(n_epochs)):
        np.random.shuffle(indexes)
        for i in tqdm(range(n_iter)):
            ib = i*batch_size
            batch_ids = indexes[ib : ib + batch_size]
            grad = np.full(N, 0, dtype = np.float)
            for k in batch_ids:
                grad[k] = der_loss(x, k, h)
            x = x - l_rate*grad
    return x

def SGD1(x0, batch_size=1, l_rate=0.01, h=0.01, n_epochs=50):
    N = len(x0)
    n_iter = N//batch_size
    indexes = np.arange(N)
    x = np.copy(x0)
    for _ in tqdm(range(n_epochs)):
        np.random.shuffle(indexes)
        for i in tqdm(range(n_iter)):
            ib = i*batch_size
            batch_ids = indexes[ib : ib + batch_size]
            grad = np.full(N, 0, dtype = np.float)
            #lx = loss(x)
            W, H = vec_to_mat0(x)
            for k in batch_ids:
                grad[k] = (loss_part(W, H, k, h) - loss_part(W, H, k, -h))/(2*h)
            x = x - l_rate*grad
    return x

def SGD(x0, batch_size=1, l_rate=0.01, h=0.01, n_epochs=50):
    N = len(x0)
    n_iter = N//batch_size
    indexes = np.arange(N)
    x = np.copy(x0)
    for _ in tqdm(range(n_epochs)):
        np.random.shuffle(indexes)
        for i in tqdm(range(n_iter)):
            ib = i*batch_size
            batch_ids = indexes[ib : ib + batch_size]
            grad = np.full(N, 0, dtype = np.float)
            grad[batch_ids] += der_loss(x, batch_ids, h)
            x = x - (l_rate/batch_size)*grad
    return x

# %% Run SGD
vec = np.copy(x0)
vec = SGD(vec, batch_size=225, l_rate=0.01, n_epochs=3)
loss(vec)
Z_sgd = vec_to_mat(vec)
print('RMSE for matrix Z_sgd: ', RMSE(Z_sgd))

#W, H = vec_to_mat(vec)
#Z_sgd = W @ H

vec1 = np.copy(vec)
loss(vec1)
RMSE(vec_to_mat(vec1))

# %% TODO Create matrix Z_perc: wypełnianie oceną odpowiadającą percentylem oceny filmu ocenie użytkownika

# %% TODO Create matrix Z_avg2 średnia po filmie i po użytkowniku 2 sposoby


# %% Program
if __name__=='__main__':
    test_alg(Z_avg_user_movie, args.alg, r = 15, i = 3, log = True)