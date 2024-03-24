import numpy as np
from sklearn.model_selection import train_test_split


def splitting(dataset, valid_size=0.1, test_size=0.0):
    """
    Split the ground truth array into training , validation , and test sets.

   Args:
     dataset: (numpy array) the ground truth mapping numpy array
     valid_size: (float) the validation set ratio
     test_size: (float) the test set ratio
   Returns:
     train_set: (numpy array) dataset for training
     val_set: (numpy array) dataset for validation
     test_set: (numpy array) dataset for testing
    """
    train_set, valid_set = train_test_split(dataset, test_size=valid_size)
    if test_size > 0:
        train_set, test_set = train_test_split(train_set, test_size=test_size)
    else:
        test_set = None

    return train_set, valid_set, test_set
