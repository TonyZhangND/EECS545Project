import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


set1 = ('../data/ArOpt.csv', '../data/ArOptLabel.csv')
set2 = ('../data/ArOpt15.csv', '../data/ArOpt15Label.csv')
set3 = ('../data/Ar15.csv', '../data/Ar15Label.csv')
set4 = ('../data/Ar25.csv', '../data/Ar25Label.csv')
all = (set1, set2, set3, set4)
features_to_use = np.array(range(15))
np.random.shuffle(features_to_use)
features_to_use = features_to_use[:3]


def generate_data_matrix_full(data):
    X = np.loadtxt(data[0], delimiter=',')
    y = np.loadtxt(data[1], delimiter=',')

    # Reduce the number of dimensions, pick some random columns
    print(f"final {X.shape}")
    return X/y.max(), y/y.max()


def generate_data_matrix_random_features(data, cols):
    X = np.loadtxt(data[0], delimiter=',')
    y = np.loadtxt(data[1], delimiter=',')

    # Reduce the number of dimensions, pick some random columns
    columns = [X[:, c] for c in cols]
    X = np.column_stack(columns)
    print(f"final {X.shape}")
    return X/y.max(), y/y.max()


def generate_data_matrix_PCA(data, components):
    X = np.loadtxt(data[0], delimiter=',')
    y = np.loadtxt(data[1], delimiter=',')
    pca = PCA(n_components=components)
    pca.fit(X)
    X = pca.transform(X)
    print(f"final {X.shape}")
    return X/y.max(), y/y.max()


def generate_data_matrix_all_PCA():
    data = [np.loadtxt(s[0], delimiter=',') for s in all]
    labels = [np.loadtxt(s[1], delimiter=',') for s in all]
    X = np.vstack(data)
    y = np.concatenate(labels, axis=None)
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    return X/y.max(), y/y.max()


def generate_data_matrix_all():
    data = [np.loadtxt(s[0], delimiter=',') for s in all]
    labels = [np.loadtxt(s[1], delimiter=',') for s in all]
    X = np.vstack(data)
    y = np.concatenate(labels, axis=None)
    return X, y


def xgbTrain(X, y):
    """
    Partitions data into a train and test set, and trains an XGBoost model given training data.
    :param X: Training data, 2D numpy array
    :param y: Training labels, 1D numpy array
    :return: Trained Booster, train rmse list, test tmse list
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print(f"train shape X:{X_train.shape}, y:{y_train.shape}")
    xgdmat = xgb.DMatrix(X_train, label=y_train)
    testmat = xgb.DMatrix(X_test, label=y_test)

    progress = dict()
    watchlist = [(xgdmat, 'train'), (testmat, 'test')]

    param = {'max_depth':6, 'eta':0.01, 'verbosity':1, 'objective':'reg:squarederror', 'lambda':1}

    gb = xgb.train(param, xgdmat, 1000, watchlist, evals_result=progress)
    testmat = xgb.DMatrix(X_test)
    y_pred = gb.predict(testmat)
    # print(f"prediction {y_pred}")
    # print(f"truth {y_test}")
    # print(progress)
    return gb, progress['train']['rmse'], progress['test']['rmse']


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
    for s in all:
        # X, y = generate_data_matrix_random_features(s, features_to_use)
        # X, y = generate_data_matrix_full(s)
        X, y = generate_data_matrix_PCA(s, 5)
        # X, y = generate_data_matrix_all_PCA()
        gb, train_error, test_error = xgbTrain(X, y)
        plot_results(train_error, test_error, title=s[0])


if __name__ == "__main__":
    main()