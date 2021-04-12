# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:59:05 2021

@author: Wojtek
"""
# %% Import
import matplotlib.pyplot as plt

# %% Tests
Zs = {'Z_avg': Z_avg,'Z_avg_movie': Z_avg_movie,'Z_avg_user': Z_avg_user}
for name, Z in Zs.items():
    print(f'SVD1 - {name} - {test_SVD1(Z)}')
    print(f'NMF  - {name} - {test_NMF(Z)}')
    print(f'SVD2 - {name} - {test_SVD2(Z)}')

# %%
def Z_mean(x):
    M = x*Z_avg_user + (1-x)*Z_avg_movie
    return RMSE(M)

# %% Lambda plot
xs = np.linspace(0,1,20)
ys = np.vectorize(Z_mean)(xs)

plt.plot(xs, ys)
plt.xlabel('lambda')
plt.ylabel('RMSE')