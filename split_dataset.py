import pandas as pd
from sklearn.model_selection import train_test_split as tts
import argparse

parser = argparse.ArgumentParser(description = 'Split csv file into training and test datasets')
parser.add_argument('src', help = 'source file')
parser.add_argument('size', help = 'size of test dataset / size of source dataset')
args = parser.parse_args()

df = pd.read_csv(args.src)
train, test = tts(df, test_size = float(args.size), random_state = 42, shuffle = True)
name = args.src.split('.')[0] # extract name without .csv from filename
train.to_csv(name + '_train.csv', index = False)
test.to_csv(name + '_test.csv', index = False)