import pandas as pd
from sklearn.model_selection import train_test_split as tts
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Split csv file into training and test datasets')
    parser.add_argument('--src', help = 'source file', default = 'data/ratings.csv')
    parser.add_argument('--size', help = 'size of test dataset / size of source dataset', default = 0.1)
    args = parser.parse_args()
    df = pd.read_csv(args.src)
    train, test = tts(df, test_size = float(args.size), random_state = 42, shuffle = True, stratify = df['userId'])
    name = args.src.split('.')[0] # extract name without .csv from filename
    train_name = name + '_train.csv'
    test_name = name + '_test.csv'
    train.to_csv(train_name, index = False)
    test.to_csv(test_name, index = False)
    print(f'Dataset split into files: {train_name}, {test_name}.')

if __name__ == '__main__':
    parse_arguments()