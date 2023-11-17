"""
This is the provided pseudo-code and the template that you can implement the regression with different order
 of polynomial regression as well as the evaluation of cross-validation. Last, you can also visualize the
 polynomial regression residual error figure

MyRegression takes the X, y ,split and order as input and return the error_dict that contain the mse of different fold
 of the dataset

VisualizeError is used to plot the figure of the error analysis
"""
# Hints
# the numpy package is useful for calculation
# sklearn.linear_model is another useful tool that you can use to fit the model with the data
# refer https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html to see how to use it
# poly.fit_transform also a useful function to generate the X matrix that can be the input of LinearRegression
# refer https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html to see how to use it


# Header
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Implement the Perceptron algorithm
def MyRegression(X, y, split, order = 2):
    # initialize the error dict, where the key is the k-th fold, and the error_dict[k] is the mean square error of
    # the test sample in the k-th fold.
    error_dict = {}
    
    for k in range(10):
        # Initialize training and test arrays
        X_train = np.zeros((int(len(X)*0.9),1))
        X_test  = np.zeros((int(len(X)*0.1),1))
        y_train = np.zeros((int(len(X)*0.9),1))
        y_test  = np.zeros((int(len(X)*0.1),1))
        
        train = 0;
        test  = 0;
        for i in range(len(X)):
            # Select test set
            if split[i] == k:
                X_test[test][0] = X[i]
                y_test[test][0] = y[i]
                test += 1
            # Select training set
            else:
                X_train[train][0] = X[i]
                y_train[train][0] = y[i]
                train += 1
                
        # Train a polynomial regression
        poly = PolynomialFeatures(degree=order)
        nX_train = poly.fit_transform(X_train)
        lm = LinearRegression()
        reg = lm.fit(nX_train, y_train)
        
        # Predict and calculate errors
        nX_test = poly.fit_transform(X_test)
        predict = reg.predict(nX_test)
        ground_truth = y_test
        mse = ((predict - ground_truth) ** 2).mean()
        
        # Save error for every fold
        error_dict[k] = mse
        
    return error_dict


def VisualizeError(error_related_to_order):
    pass
    # Initialize order and error arrays
    order = []
    errors = []
    
    # Collect order and values from error dictionary
    for key, values in error_related_to_order.items():
        order.append(key)
        errors.append(np.mean(values))
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(order, errors, '--ro')
    ax.set_title('MSE Using Different Orders')
    ax.set_xlabel('Order of Polynomial Regression')
    ax.set_ylabel('Mean of MSE')
    ax.set_xlim([1.5,6.5])
    ax.set_ylim([-20,150])
    ax.set_xticks(np.arange(2, 7, 1))
    plt.savefig('images/prob6.jpg')
    plt.show()



