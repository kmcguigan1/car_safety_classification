# imports of default python modules

# other dependency imports
import pandas as pd
import numpy as np

# sklearn imports
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Constants
DATASET_INFO_PATH = "Dataset\\car.c45-names"
DATASET_PATH = "Dataset\\car.data"

# Hyper Parameters
TRAIN_SIZE = 0.8            # size of the train split in the knn
SEED = 321                  # random seed to use throught the program
K = 8                       # the number of neighbors for the knn to consider
WEIGHT_TYPE = "distance"    # could also be "uniform" this is how the knn uses neighbors, either in a uniform way or inversely proportional to their distance from the point of interest

def read_info_file(path):
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

def get_encoded_data():
    categories = read_info_file(DATASET_INFO_PATH)
    data = read_csv(DATASET_PATH)
    encoders = {}
    for col in data.columns:
        # create an encoder for this column
        encoder = OrdinalEncoder(categories=[categories[col]], dtype=int)
        # fit the encoder to the data and transform the data using the encoder
        data[col] = encoder.fit_transform(np.reshape(data[col].values, newshape=(-1,1)))
        # print the encoder categories
        print(f"{col}: {encoder.categories_[0]}")
        # save the encoder for later use in predictions
        encoders[col] = encoder
    return data, encoders

def split_data(data):
    # get the list of columns in the dataframe
    columns = list(data.columns)
    # remove class form the list
    columns.remove("class")
    # get the train test split
    x_train, x_test, y_train, y_test = train_test_split(data[columns].values, data["class"].values, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
    return x_train, x_test, y_train, y_test

def get_best_model(x_train, x_test, y_train, y_test):
    # lets test out to find the most performant settings, small dataset so just brute force is fine
    best_result = {"weighting":None, "k":None, "score":0}
    best_model = None
    for weighting in ["uniform", "distance"]:
        for k in range(1, 20, 1):
            # create the model
            knn = KNeighborsClassifier(n_neighbors=k, weights=weighting, algorithm="auto")
            # fit the knn
            knn.fit(x_train, y_train)
            # test the knn
            score = knn.score(x_test, y_test)
            # output the results
            #print(f"Weighting {weighting}, K {k}, score {score}")
            # if the score is better than the current best score save this entry as the best score
            if(score > best_result["score"]):
                best_result = {"weighting":weighting, "k":k, "score":score}
                best_model = knn
            # end of for k in range loop
        # end of for weighting in list loop
    # print the best result
    print("\nBEST RESULT")
    print("==================")
    print(f"Weighting {best_result['weighting']}, K {best_result['k']}, score {best_result['score']}")
    print("==================")
    return best_model

def get_model(x_train, x_test, y_train, y_test):
    # create the model
    knn = KNeighborsClassifier(n_neighbors=K, weights=WEIGHT_TYPE, algorithm="auto")
    # fit the model
    knn.fit(x_train, y_train)
    # score the model
    score = knn.score(x_test, y_test)
    # show the score
    print(f"\nScore {score}")
    # return the model
    return knn

def test_data(data, encoders):
    return

def main():
    # get the data and the encoders used to encode the data, this would be used if we fed new data to the knn
    data, encoders = get_encoded_data()
    # get the split data
    x_train, x_test, y_train, y_test = split_data(data)
    # get the best model
    knn = get_model(x_train, x_test, y_train, y_test)
    # now lets do some graphing to see how the data relates to the classifications
    return 

if __name__=='__main__':
    main()

