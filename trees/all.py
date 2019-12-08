import numpy as np
import matplotlib.pyplot as plt
import sklearn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split
import pandas as pd
import random

from sklearn import datasets
from djinn import djinn
print(sklearn.__version__)

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

def computeMSE(x_train, x_test, y_train, y_test, feature):

    x_train[:,feature] = 0
    x_test[:,feature] = 0

    print("djinn example")    
    modelname="reg_djinn_test"   # name the model
    ntrees=1                 # number of trees = number of neural nets in ensemble
    maxdepth=4               # max depth of tree -- optimize this for each data set
    dropout_keep=1.0         # dropout typically set to 1 for non-Bayesian models

    #initialize the model
    model=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

    # find optimal settings: this function returns dict with hyper-parameters
    # each djinn function accepts random seeds for reproducible behavior
    optimal=model.get_hyperparameters(x_train, y_train, random_state=1)
    batchsize=optimal['batch_size']
    learnrate=optimal['learn_rate']
    epochs=optimal['epochs']


    # train the model with hyperparameters determined above
    model.train(x_train,y_train,epochs=epochs,learn_rate=learnrate, batch_size=batchsize, 
                  display_step=1, save_files=True, file_name=modelname, 
                  save_model=True,model_name=modelname, random_state=1)

    # *note there is a function model.fit(x_train,y_train, ... ) that wraps 
    # get_hyperparameters() and train(), so that you do not have to manually
    # pass hyperparameters to train(). However, get_hyperparameters() can
    # be expensive, so I recommend running it once per dataset and using those
    # hyperparameter values in train() to save computational time

    # make predictions
    m=model.predict(x_test) #returns the median prediction if more than one tree

    #evaluate results
    mse=sklearn.metrics.mean_squared_error(y_test,m)
    mabs=sklearn.metrics.mean_absolute_error(y_test,m)
    exvar=sklearn.metrics.explained_variance_score(y_test,m)   
    print('MSE',mse)
    print('M Abs Err',mabs)
    print('Expl. Var.',exvar)

    #close model 
    model.close_model()

    print("Reload model and continue training for 20 epochs")
    # reload model; can also open it using cPickle.load()
    model2=djinn.load(model_name="reg_djinn_test")

    #continue training for 20 epochs using same learning rate, etc as before
    model2.continue_training(x_train, y_train, 20, learnrate, batchsize, random_state=1)

    #make updated predictions
    m2=model2.predict(x_test)

    #evaluate results
    mse2=sklearn.metrics.mean_squared_error(y_test,m2)
    mabs2=sklearn.metrics.mean_absolute_error(y_test,m2)
    exvar2=sklearn.metrics.explained_variance_score(y_test,m2)   
    print('MSE',mse2)
    print('M Abs Err',mabs2)
    print('Expl. Var.',exvar2)


    # Bayesian formulation with dropout. Recommend dropout keep 
    # probability ~0.95, 5-10 trees.
    print("Bayesian djinn example")
    ntrees=10
    dropout_keep=0.95
    modelname="reg_bdjinn_test"

    # initialize a model
    bmodel=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

    # "fit()" does what get_hyperparameters + train does, in one step: 
    bmodel.fit(x_train,y_train, display_step=1, save_files=True, file_name=modelname, 
               save_model=True,model_name=modelname, random_state=1)

    # evaluate: niters is the number of times you evaluate the network for 
    # a single sample. higher niters = better resolved distribution of predictions
    niters=100
    bl,bm,bu,results=bmodel.bayesian_predict(x_test,n_iters=niters, random_state=1)
    # bayesian_predict returns 25, 50, 75 percentile and results dict with all predictions

    # evaluate performance on median predictions
    mse=sklearn.metrics.mean_squared_error(y_test,bm)
    mabs=sklearn.metrics.mean_absolute_error(y_test,bm)
    exvar=sklearn.metrics.explained_variance_score(y_test,bm)   
    print('MSE',mse)
    print('M Abs Err',mabs)
    print('Expl. Var.',exvar)

#     # make a pretty plot
#     g=np.linspace(np.min(y_test),np.max(y_test),10)    
#     fig, axs = plt.subplots(1,1, figsize=(8,8), facecolor='w', edgecolor='k')
#     fig.subplots_adjust(hspace = .15, wspace=.1)
#     sc=axs.scatter(y_test, bm, linewidth=0,s=6, 
#                       alpha=0.8, c='#68d1ca')
#     a,b,c=axs.errorbar(y_test, bm, yerr=[bm-bl,bu-bm], marker='',ls='',zorder=0, 
#                        alpha=0.5, ecolor='black')
#     axs.set_xlabel("True")
#     axs.set_ylabel("B-DJINN Prediction")    
#     axs.plot(g,g,color='red')
# #     axs.text(0.5, 0.9,filename, fontsize = 28, ha='center', va='center', transform=axs.transAxes)
#     axs.text(0.4, 0.75,'mesTest=%.5f' %mse, fontsize = 28, ha='center', va='center', transform=axs.transAxes)

#     # plt.show()
#     plt.savefig(resultPath+'allTest', bbox_inches="tight", dpi = 300)    
#     # collect_tree_predictions gathers predictions in results dict
#     # in a more intuitive way for easy plotting, etc
#     p=bmodel.collect_tree_predictions(results['predictions'])

    return mse
    
    
random.seed(2019)
resultPath = 'C:/Users/linji/OneDrive - Umich/EECS545Fall2019/Project/DJINN/'
# filename = 'All'
#Load the data, split into training/testing groups

X_array_1 = pd.read_csv('ArOpt15.csv', header=None).values
Y_array_1 = pd.read_csv('ArOpt15Label.csv', header=None).values.T

X_array_2 = pd.read_csv('Ar15.csv', header=None).values
Y_array_2 = pd.read_csv('Ar15Label.csv', header=None).values.T

X_array_3 = pd.read_csv('Ar25.csv', header=None).values
Y_array_3 = pd.read_csv('Ar25Label.csv', header=None).values.T

X_array_4 = pd.read_csv('ArOpt.csv', header=None).values
Y_array_4 = pd.read_csv('ArOptLabel.csv', header=None).values.T


X_array = np.concatenate((X_array_1, X_array_2, X_array_3, X_array_4),axis=0)
Y_array = np.concatenate((Y_array_1, Y_array_2, Y_array_3, Y_array_4),axis=0)
Y_array = Y_array/Y_array.max()


xtrain, xtest, ytrain, ytest = train_test_split(X_array, Y_array, test_size=0.2, random_state=1)
xtrain, xtest = sphere(xtrain, xtest)

    
mseArray = np.zeros(xtrain.shape[1])
for i in range(14, xtrain.shape[1]):
    mseArray[i] = computeMSE(xtrain, xtest, ytrain, ytest,i)
    print(i)
print(mseArray)

# np.savetxt(resultPath + 'featureImportance.csv', mseArray, delimiter=',', fmt='%1.5f')