import numpy as np
from sklearn.model_selection import train_test_split


def splitting(ground_truth_mapping, val_size=0.1, test_size=0.1):
    """
    Split the ground truth array into training , validation , and test sets.

   Args:
     ground_truth_mapping: (numpy array) the ground truth mapping numpy array
     val_size: (float) the validation set ratio
     test_size: (float) the test set ratio
   Returns:
     train_set: (numpy array) dataset for training
     val_set: (numpy array) dataset for validation
     test_set: (numpy array) dataset for testing
    """
    x_train, x_test, y_train, y_test = train_test_split(ground_truth_mapping, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, test_size=val_size)

    train_set = x_train, y_train
    val_set = x_val, y_val
    test_set = x_test, y_test

    return train_set, val_set, test_set
