import matplotlib
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import operator
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


def read_data(datasets):
    """
    :param datasets: list containing (X.csv, y.csv) pairs
    :return: pair (X, y) where X is the feature sample matrix and y is the label array from combining all data
        in datasets
    """
    data = [np.loadtxt(s[0], delimiter=',') for s in datasets]
    labels = [np.loadtxt(s[1], delimiter=',') for s in datasets]
    X = np.vstack(data)
    y = np.concatenate(labels, axis=None)
    return X, y


def sphere(X_train, X_test):
    """
    :param X_train: Sample-feature matrix to sphere
    :param X_test: Sample-feature matrix to sphere according to mean and stdev of X_train
    :return: Tuple containing (X_train_sphered, X_test_sphered)
    """
    X_train, X_test = X_train.T, X_test.T
    a, b = X_train.shape
    stdevs = [np.std(row) for row in X_train]  # standard deviation of each row in X
    diag = np.diag([1 / s for s in stdevs])
    X_train_sphered = diag.dot(X_train).dot(np.eye(b) - 1 / b * np.ones((b, b)))
    sample_means = np.array([np.mean(row) for row in X_train])
    sample_stds = np.array([np.std(row) for row in X_train])

    # Now update X_test according to sample_means, sample_stds
    a, b = X_test.shape
    # print(f"X_test shape {X_test.shape}")
    # print(f"means.shape {sample_means.shape}, stds.shape {sample_stds.shape}")
    assert sample_stds.shape[0] == a and sample_means.shape[0] == a
    X_test_sphered = X_test - np.column_stack([sample_means for i in range(b)])
    X_test_sphered = X_test_sphered / np.column_stack([sample_stds for i in range(b)])
    return X_train_sphered.T, X_test_sphered.T


def process_data(X, y, test_size=0.2, random_state=42):
    """
    :param X: 2D numpy array containing all training samples
    :param y: 1D numpy array containing all training labels corresponding to X
    :param test_size: fraction of samples to use for training
    :param random_state: seed for sklearn.model_selection.train_test_split
    :return: (X_train, X_test, y_train, y_test) tuple
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y/y.max(), test_size=test_size, random_state=random_state)
    X_train, X_test = sphere(X_train, X_test)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    return X_train, X_test, y_train, y_test


def xgbTrain(X_train, X_test, y_train, y_test, my_params):
    """
    Trains an XGBoost model given training data.
    :param X_train: Training sample-feature matrix
    :param X_test: Testing sample-feature matrix
    :param y_train: Training labels
    :param y_test: Testing labels
    :param my_params: parameters for XGBoost
    :return: Trained Booster, train rmse list, test rmse list
    """
    xgdmat = xgb.DMatrix(X_train, label=y_train)
    testmat = xgb.DMatrix(X_test, label=y_test)

    # Train final booster
    progress = dict()
    watchlist = [(xgdmat, 'train'), (testmat, 'test')]
    gb = xgb.train(my_params, xgdmat, 20000, watchlist, evals_result=progress, early_stopping_rounds=50)
    testmat = xgb.DMatrix(X_test)
    y_train_pred = gb.predict(xgdmat)
    y_test_pred = gb.predict(testmat)
    return gb, progress['train']['rmse'], progress['test']['rmse'], y_train_pred, y_test_pred


def nFoldCV(n, cv_params, ind_params, X_train, y_train):
    """
    Performs cross-validation to tune the hyperparameters of XGBoost
    :param n: Number of cv folds
    :param cv_params: Parameters to do grid search
    :param ind_params: Fixed parameters
    :param X_train: Training sample-feature matrix
    :param y_train: Training labels
    :return: Best parameters for XGBoost
    """
    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                 cv_params,
                                 scoring='neg_mean_squared_error', cv=n, n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    return optimized_GBM.best_params_


def plot_results(train_error, test_error, title=None):
    legends = ('train', 'test')
    plt.plot(list(range(len(train_error))), train_error, color='blue')
    plt.plot(list(range(len(test_error))), test_error, color='red')
    # plt.yscale('log')
    if title:
        plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('mse')
    plt.legend(legends, loc='upper right')
    plt.axes().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    # plt.grid(which='both', axis='both')
    plt.show()


def plot(y, yTestReal, prediction, y_predict):
    xx = np.linspace(0, 1, 100)
    yy = np.linspace(0, 1, 100)
    plt.clf()
    plt.plot(xx, yy, label='y=x')
    plt.xlabel('y_real')
    plt.ylabel('y_predict')
    plt.plot(y, prediction, '.', label='train')
    plt.plot(yTestReal, y_predict, 'o', label='test')
    plt.legend()
    plt.show()


def do_importance(m):
    sorted_m = sorted(m.items(), key=operator.itemgetter(1))
    sorted_m.reverse()
    print(sorted_m)



def main():
    set1 = ('../data/ArOpt.csv', '../data/ArOptLabel.csv')
    set2 = ('../data/ArOpt15.csv', '../data/ArOpt15Label.csv')
    set3 = ('../data/Ar15.csv', '../data/Ar15Label.csv')
    set4 = ('../data/Ar25.csv', '../data/Ar25Label.csv')
    datasets = (set1, set2, set3, set4)

    X, y = read_data(datasets)
    X_train, X_test, y_train, y_test = process_data(X, y, test_size=0.2, random_state=42)
    # cv_params = {'max_depth': [9],
    #              'min_child_weight': [9],
    #              'subsample': [0.9],
    #              'colsample_bytree': [0.9],
    #              'learning_rate': [0.1]
    #              }
    cv_params1 = {'max_depth': [5, 7, 9],
                  'min_child_weight': [5, 7, 9],
                  'subsample': [0.7, 0.8, 0.9],
                  'colsample_bytree': [0.7, 0.8, 0.9],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'alpha': [1, 10, 20]
                  }
    ind_params = {
        # 'learning_rate': 0.1,
                  'n_estimators': 1000,
                  'seed': 0,
                  # 'subsample': 0.8,
                  # 'colsample_bytree': 0.8,
                  'objective': 'reg:squarederror'
                  }
    print("Performing cross validation")
    # my_params = nFoldCV(5, cv_params1, ind_params, X_train, y_train)
    my_params = {'alpha': 1, 'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 5, 'subsample': 0.8}
    print(f"CV results {my_params}")
    for (k, v) in ind_params.items():
        my_params[k] = v
    print("Training XGBoost using cv results")
    gb, train_rmse, test_rmse, y_train_pred, y_test_pred = xgbTrain(X_train, X_test, y_train, y_test, my_params)
    do_importance(gb.get_score(importance_type='gain'))
    train_mse = [x ** 2 for x in train_rmse]
    test_mse = [x ** 2 for x in test_rmse]
    plot_results(train_mse, test_mse, title=None)
    plot(y_train, y_test, y_train_pred, y_test_pred)


if __name__ == "__main__":
    main()