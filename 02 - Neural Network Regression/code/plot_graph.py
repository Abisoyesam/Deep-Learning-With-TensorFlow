import matplotlib.pyplot as plt

def plot_predictions(X_train=X_train ,y_train=y_train, X_test=X_test, y_test = y_test,y_pred=y_pred):
    """
     plot training data, test data and compares prediction to ground truth
    """
    plt.figure(figsize=(10,7))
    # plot training data in blue
    plt.scatter(X_train, y_train, c="b", label="Training data")
    # plot testing data in green
    plt.scatter(X_test, y_test, c="g", label="Testing data")
    # plot model prediction
    plt.scatter(X_test, y_pred, c="r", label="Prediction")
    # show legend
    plt.legend();