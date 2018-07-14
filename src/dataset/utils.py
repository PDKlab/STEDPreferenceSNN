import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def train_valid_test_split(X, y, valid_size=0.1, test_size=0.1, random_state=42):
    train_size = 1 - test_size
    true_valid_size = valid_size / train_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state,
                                                        shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                          test_size=true_valid_size,
                                                          random_state=random_state+1,
                                                          shuffle=True)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def flatten_dataset(X, y, random_state=42):
    data = []
    for xx, yy in zip(X, y):
        p = xx[yy]
        for i in range(len(xx)):
            if i != yy:
                data.append([p, xx[i]])
    return shuffle(np.array(data), random_state=random_state)