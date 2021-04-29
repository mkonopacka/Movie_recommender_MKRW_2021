from split_dataset import parse_arguments
from recom_system_307915_300801 import *

def run_test(alg, **kwargs):
    ''' Run test on algorithm `alg` with parameters passed as keyword arguments; Returns obtained RMSE '''
    algs = {"NMF": approx_NMF,"SVD1": approx_SVD1,"SVD2": approx_SVD2,"SGD": approx_SGD}
    print('Building a model ...')
    Z_approx, log = algs[alg](**kwargs)
    result = RMSE(Z_approx)
    print(log, file = open('results_log.txt', 'a')) # append mode
    return result

if __name__ == '__main__':
    for alg in ["NMF", "SVD1", "SVD2"]:
        run_test(alg, Z_ = Z_avg_user_movie)
        run_test(alg, Z_ = Z_avg_user)
        run_test(alg, Z_ = Z_avg_movie)
        run_test(alg, Z_ = Z_close_users)

    
