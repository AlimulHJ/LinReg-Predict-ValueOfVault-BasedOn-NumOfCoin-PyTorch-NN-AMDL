import torch
import pytest
from regression import fit_regression_model

# Function for getting the training data
def get_train_data(dim=1):
    """
    Training data (Feature/Input) for the regression model.
    dim - number of features in the input. For our purposes it will be either 1 or 2.
    """
    X_2 = torch.tensor(
        [[24.,  2.],
         [24.,  4.],
         [16.,  3.],
         [25.,  6.],
         [16.,  1.],
         [19.,  2.],
         [14.,  3.],
         [22.,  2.],
         [25.,  4.],
         [12.,  1.],
         [24.,  7.],
         [19.,  1.],
         [23.,  7.],
         [19.,  5.],
         [21.,  3.],
         [16.,  6.],
         [24.,  5.],
         [19.,  7.],
         [14.,  4.],
         [20.,  3.]])

    """The target values (Value of vault) for the input data"""
    y = torch.tensor(
        [[1422.4000],
         [1469.5000],
         [1012.7000],
         [1632.2000],
         [952.2000],
         [1117.7000],
         [906.2000],
         [1307.3000],
         [1552.8000],
         [686.7000],
         [1543.4000],
         [1086.5000],
         [1495.2000],
         [1260.7000],
         [1288.1000],
         [1111.5000],
         [1523.1000],
         [1297.4000],
         [946.4000],
         [1197.1000]])
    if dim == 1:
        X = X_2[:, :1] # Extract the first column of X_2
    elif dim == 2:  # Return the whole X_2
        X = X_2
    else:
        raise ValueError("dim must be 1 or 2")
    return X, y


## Test cases:

# Test to check if loss < 4321
def test_fit_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    print(loss)

    assert loss.item() < 4321,  " loss too big"


# Test to check if loss < 400
def test_fit_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    assert loss.item() < 400


# Test to check if the model is predicting the correct values for the 1D input data
def test_fit_and_predict_regression_model_1d():
    X, y = get_train_data(dim=1)    # Get the training data
    model, loss = fit_regression_model(X, y)    # Fit the model
    X_test = torch.tensor([[20.], [15.], [10.]])    # Test data
    y_pred = model(X_test)  # Predict the values

    # Check if the predicted values are correct
    assert ((y_pred - torch.tensor([[1252.3008],
                                    [939.9971],
                                    [627.6935]])).abs() < 2).all(), " y_pred is not correct"

    # Check if the shape of the predicted values are correct
    # The shape should be (3, 1) // 3 - number of samples & 1 - number of features
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"


# Test to check if the model is predicting the correct values for the 2D input data
def test_fit_and_predict_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20., 2.], [15., 3.], [10., 4.]])
    y_pred = model(X_test)

    assert ((y_pred - torch.tensor([[1191.9037],
                                    [943.9369],
                                    [695.9700]])).abs() < 2).all(), " y_pred is not correct"
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"


# Run the test cases from this main function
# Run this file using the command: python regression_test.py
# If test cases = pass >> output will be: "All test cases passed"
# If test cases = fail >> output will be: the error message
if __name__ == "__main__":
    test_fit_regression_model_1d()
    test_fit_regression_model_2d()
    test_fit_and_predict_regression_model_1d()
    test_fit_and_predict_regression_model_2d()
