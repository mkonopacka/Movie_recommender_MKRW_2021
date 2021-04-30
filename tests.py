from split_dataset import parse_arguments
from recom_system_307915_300801 import *

if __name__ == '__main__':
    # mat_names = {
    #     "Z_avg_user_movie": Z_avg_user_movie,
    #     "Z_avg_user": Z_avg_user,
    #     "Z_avg_movie": Z_avg_movie
    # }
    # for name in mat_names:
    #     for alg in ["NMF", "SVD1", "SVD2"]:
    #         run_test(alg, mat_name = name, Z_ = mat_names[name])
    
    run_test("SGD", mat_name= 'Z_avg_user_movie')