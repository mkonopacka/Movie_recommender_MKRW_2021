from split_dataset import parse_arguments
from recom_system_307915_300801 import *

if __name__ == '__main__':
    run_test("NMF2_SVD2", Z_ = Z_avg_user_movie)
    # mat_names = {
    #     "Z_avg_user_movie": Z_avg_user_movie,
    #     "Z_avg_user": Z_avg_user,
    #     "Z_avg_movie": Z_avg_movie
    # }
    # for name in mat_names:
    #     for alg in ["NMF", "SVD1", "SVD2"]:
    #         run_test(alg, mat_name = name, Z_ = mat_names[name])
    
    # run_test("SGD", mat_name= 'Z_avg_user_movie')
    # run_test("NMF", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 3)
    # run_test("NMF", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 5)
    # run_test("SVD1", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 5)
    # run_test("SVD1", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 10)
    # run_test("SVD2", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 5, i = 7)
    # run_test("SVD2", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 5, i = 10)
    # run_test("SVD2", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 3, i = 3)
    # run_test("SVD2", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 3, i = 5)
    # run_test("SVD2", "Z_avg_user_movie", Z_ = Z_avg_user_movie, r = 3, i = 10)