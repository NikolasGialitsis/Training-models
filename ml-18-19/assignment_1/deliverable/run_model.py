import joblib
import numpy as np


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    return ((y_true - y_pred) ** 2).mean()


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x_test, y_test = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the trained model
    import joblib   
    from keras import models
    import tensorflow as tf
    import keras
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    while True:
        nb = input('Instructions\nPress 1 to run the linear_model\nPress 2 to run the non_linear model: ')
        if nb == '1':
            print('\tRunning linear model selected')
            p = PolynomialFeatures(interaction_only=True,include_bias = False,degree=2)
            x_test = p.fit_transform(x_test)
            model = load_model('linear_model.pickle')        
            break
        elif nb == '2':
            x_test =StandardScaler().fit_transform(x_test)
            sklearn_pca=PCA(n_components=1)
            x_test=sklearn_pca.fit_transform(x_test)

            print('\tRunning non_linear model selected')
            model = models.load_model('non_linear_model.pickle')
            break
        else:
            print('\n\nplease try again')
            continue

    y_pred = model.predict(x_test)


    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y_test)
    print('MSE: {}'.format(mse))
