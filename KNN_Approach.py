# imports of default python modules

# other dependency imports
import pandas as pd
import numpy as np

# sklearn imports
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Hyper Parameters
TRAIN_SIZE = 0.8

def read_file(path):
    data = {}
    for idx, line in enumerate(open(path, mode="r")):
        if(idx == 4):
            line = line.replace(" ", "").replace("\n", "")
            data['class'] = line.split(',')
        elif(idx >= 8):
            line = line.replace(" ", "").replace("\n", "")
            line_info = line.split(':')
            data[line_info[0]] = line_info[1].split(',')
    return data

def read_csv(path):
    data = pd.read_csv(path)
    return data

categories = read_file("Dataset\\car.c45-names")
data = read_csv("Dataset\\car.data")
encoders = {}
for col in data.columns:
    # create an encoder for this column
    encoder = OrdinalEncoder(categories=[categories[col]], dtype=int)
    # fit the encoder to the data and transform the data using the encoder
    data[col] = encoder.fit_transform(np.reshape(data[col].values, newshape=(-1,1)))
    # print the encoder categories
    print(encoder.categories_)
    # save the encoder for later use in predictions
    encoders[col] = encoder
print(data.head())

# get the list of columns in the dataframe
columns = list(data.columns)
# remove class form the list
columns.remove("class")
# get the train test split
x_train, x_test, y_train, y_test = train_test_split(data[columns].values, data["class"].values, train_size=TRAIN_SIZE)
print(len(x_train), len(x_test), len(y_train), len(y_test))
