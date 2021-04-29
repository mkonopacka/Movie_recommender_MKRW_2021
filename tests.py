from numpy.matrixlib.defmatrix import matrix
from split_dataset import parse_arguments
from recom_system_307915_300801 import *
from time import perf_counter as time

def run_test(alg, mat_name, **kwargs):
    ''' Run test on algorithm `alg` with parameters passed as keyword arguments; Returns obtained RMSE '''
    algs = {"NMF": approx_NMF,"SVD1": approx_SVD1,"SVD2": approx_SVD2,"SGD": approx_SGD}
    print('Building a model ...')
    start = time()
    Z_approx, log = algs[alg](**kwargs)
    result = RMSE(Z_approx)
    stop = time()
    print(f'For matrix {mat_name} ' + log + f' (Total time: {stop - start})', file = open('results_log.txt', 'a')) # append mode
    return result

if __name__ == '__main__':
    matrix_names = {
        "Z_avg_user_movie" : Z_avg_user_movie,
        "Z_avg_user": Z_avg_user,
        "Z_avg_movie": Z_avg_movie,
        "Z_close_users": Z_close_users
    }

    for alg in ["NMF", "SVD1", "SVD2"]:
        for name in matrix_names:
            run_test(alg, name, Z_ = matrix_names[name])