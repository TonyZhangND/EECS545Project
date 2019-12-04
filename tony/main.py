import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


def xgbTrain(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost model given training data.
    :param X_train: Training sample-feature matrix
    :param X_test: Testing sample-feature matrix
    :param y_train: Training labels
    :param y_test: Testing labels
    :return: Trained Booster, train rmse list, test rmse list
    """
    xgdmat = xgb.DMatrix(X_train, label=y_train)
    testmat = xgb.DMatrix(X_test, label=y_test)

    progress = dict()
    watchlist = [(xgdmat, 'train'), (testmat, 'test')]

    param = {'max_depth':6, 'eta':0.1, 'verbosity':1, 'objective':'reg:squarederror', 'lambda':1}

    gb = xgb.train(param, xgdmat, 100, watchlist, evals_result=progress)
    testmat = xgb.DMatrix(X_test)
    y_pred = gb.predict(testmat)
    # print(f"prediction {y_pred}")
    # print(f"truth {y_test}")
    # print(progress)
    return gb, progress['train']['rmse'], progress['test']['rmse']


def xgbTrainCV(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost model given training data.
    :param X_train: Training sample-feature matrix
    :param X_test: Testing sample-feature matrix
    :param y_train: Training labels
    :param y_test: Testing labels
    :return: Trained Booster, train rmse list, test rmse list
    """
    xgdmat = xgb.DMatrix(X_train, label=y_train)
    testmat = xgb.DMatrix(X_test, label=y_test)

    cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'reg:squarederror'}
    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                 cv_params,
                                 scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    print(optimized_GBM.error_score)

    # progress = dict()
    # watchlist = [(xgdmat, 'train'), (testmat, 'test')]
    #
    # param = {'max_depth':6, 'eta':0.1, 'verbosity':1, 'objective':'reg:squarederror', 'lambda':1}
    #
    # cv = xgb.cv(params=param, dtrain=xgdmat, num_boost_round=3000, nfold=5, metrics=['error'], early_stopping_rounds=100)
    # print(cv)
    # testmat = xgb.DMatrix(X_test)
    # y_pred = gb.predict(testmat)
    # # print(f"prediction {y_pred}")
    # # print(f"truth {y_test}")
    # # print(progress)
    # return gb, progress['train']['rmse'], progress['test']['rmse']


def plot_results(train_error, test_error, title=None):
    legends = ('train', 'test')
    plt.plot(list(range(len(train_error))), train_error)
    plt.plot(list(range(len(test_error))), test_error)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('rmse')
    plt.legend(legends, loc='upper right')
    # plt.axes().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    # plt.axes().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    # plt.axes().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100000))
    plt.grid(which='both', axis='both')
    plt.show()


def main():
    set1 = ('../data/ArOpt.csv', '../data/ArOptLabel.csv')
    set2 = ('../data/ArOpt15.csv', '../data/ArOpt15Label.csv')
    set3 = ('../data/Ar15.csv', '../data/Ar15Label.csv')
    set4 = ('../data/Ar25.csv', '../data/Ar25Label.csv')
    datasets = (set1, set2, set3, set4)

    X, y = read_data(datasets)
    X_train, X_test, y_train, y_test = process_data(X, y, test_size=0.2, random_state=42)
    gb, train_rmse, test_rmse = xgbTrainCV(X_train, X_test, y_train, y_test)
    train_mse = [x ** 2 for x in train_rmse]
    test_mse = [x ** 2 for x in test_rmse]
    # plot_results(train_mse[30:], test_mse[30:], title="Plot")


if __name__ == "__main__":
    main()