'''
==============================
@Title  Regression assignment 
@Course Machine learning 2019
@Author Nikolaos Gialitsis
=============================

'''
#basic libraries
import numpy as np
import pickle
import numpy
import pandas
import scipy
import joblib
import keras
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from run_model import * #user library
#Deep learning modules
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import tensorflow as tf





filename = '../data/data.npz'
data = np.load(filename)
X = data['x']
Y = data['y']
print('size of X = {}'.format(X.shape))
print('size of Y = {}'.format(Y.shape))

#normalize dataset
X =StandardScaler().fit_transform(X)


'''
=================================================================================================
                             L I N E A R     M O D E L
=================================================================================================
'''

print('==================================================================================')
print('======================     LINEAR REGRESSION  ====================================')
lr = LinearRegression() 
k = 10
kf = KFold(n_splits=k, shuffle=False)
scores_lr = []
errors_lr = []
print('[Linear Regression] running 10-fold cross-validation')
for train_indices, val_indices in kf.split(X, Y): #split into training and test set

  X_train = X[train_indices]
  X_test = X[val_indices]
  Y_train = Y[train_indices]
  Y_test = Y[val_indices]

  #transform to format f(x,theta) = theta0 + theta1*x1 + theta2*x2 + theta3*x1*x2
  p = PolynomialFeatures(interaction_only=True,include_bias = False,degree=2)
  X_train = p.fit_transform(X_train)
  X_test = p.fit_transform(X_test)
  poly_feature_names = p.get_feature_names()

  #train model
  lr.fit(X_train,Y_train)

  #model parameters
  beta_est = [lr.coef_[0], lr.intercept_]
  intercept = lr.intercept_
  coefficients = lr.coef_ #theta values
  assert(len(coefficients) == len(poly_feature_names))

  #predict values for testing set
  Y_est = lr.predict(X_test)

  #calculate Mean Square error and accuracy
  MSE = evaluate_predictions(y_true= Y_test,y_pred=Y_est)
  errors_lr.append(MSE)
  score = lr.score(X_test, Y_test)
  scores_lr.append(score)


print('[Linear Regression] (Cross-Validation) Mean accuracy = ',abs(np.mean(scores_lr)))
print('[Linear Regression] (Cross-Validation) Mean MSE = ',abs(np.mean(errors_lr)))


p = PolynomialFeatures(interaction_only=True,include_bias = False,degree=2)
Xpol = p.fit_transform(X)
poly_feature_names = p.get_feature_names()
#partition into training and validation set
(x_pol_train, x_pol_test,y_train,y_test) = train_test_split(Xpol,Y, test_size=0.1, random_state=40)

#train model
lr.fit(x_pol_train,y_train)

#model parameters
beta_est = [lr.coef_[0], lr.intercept_]
learned_f = 'learned function from regression: '
intercept = lr.intercept_
learned_f += '{:+.3f}'.format(intercept)
coefficients = lr.coef_
assert(len(coefficients) == len(poly_feature_names))

#print learned function
for i in range(0, len(coefficients)):
    learned_f += ' {:+.3f} {}'.format(coefficients[i], poly_feature_names[i])
print(learned_f)

#predict values
y_test_est = lr.predict(x_pol_test)
MSE = evaluate_predictions(y_true= y_test,y_pred=y_test_est)
print('[Linear Regression] MSE on test set = ',MSE)

score=r2_score(y_true=y_test,y_pred=y_test_est)
print('[Linear Regression] Accuracy on test set:   {:.3f}'.format(score))
print('==================================================================================')
print('==================================================================================\n\n')

#save sklearn model into a .pickle format
joblib.dump(lr, 'linear_model.pickle')

'''
#plot estimated vs true functions
fig = plt.figure(1)
plt.title('Model Predictions')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(Xpol[0],Xpol[1],y_test , label = 'Predicting test performance');
ax.scatter(Xpol[0],Xpol[1],y_test_est , label = 'Estimated Values');
ax.plot_trisurf(Xpol[0],Xpol[1],x_pol_test[1],y_test, alpha=0.3, label='real fun');
ax.plot_trisurf(Xpol[0],Xpol[1],x_pol_test[1],y_test_est, alpha=0.3, label='est fun');
plt.show()
'''

'''
=================================================================================================
              N O N  -  L I N E A R     M O D E L
=================================================================================================
'''


def learning_model(): #Neural Network Architecture#
  global input_dim
  model = Sequential()
  model.add(Dense(512, input_dim= input_dim,activation="relu"))
  model.add(Dense(256,  activation="relu")) #hidden layers
  model.add(Dense(200,  activation="relu"))
  model.add(Dense(128,  activation="relu"))
  model.add(Dense(100,  activation="relu"))
  model.add(Dense(64,  activation="relu"))
  model.add(Dense(32,  activation="relu"))
  model.add(Dense(16,  activation="relu"))
  model.add(Dense(8,  activation="relu"))
  model.add(Dense(4,  activation="relu"))
  model.add(Dense(2,  activation="relu"))
  model.add(Dense(1,activation='linear'))
  model.compile(loss="mean_squared_error", optimizer="adam",metrics=['accuracy'])
  return model

# fix random seed for reproducibility

epochs = 2000
seed = 40
batches = 32

print('==================================================================================')
print('====================   NON LINEAR REGRESSION   ===================================')

print("epochs = %d , batches = %d " % (epochs,batches))
np.random.seed(seed) #remove ambiguity



X = data['x']
Y = data['y']
#normalize
X =StandardScaler().fit_transform(X)

print('[Neural Network] running 10-fold cross validation')
input_dim = 2
estimator = KerasRegressor(build_fn=learning_model, epochs=500, batch_size=batches, verbose= 0)
kfold = KFold(n_splits=10, random_state=seed)
NN_score = cross_val_score(estimator, X, Y, cv=kfold)
print('[Neural Network] (Cross-Validation) Mean score [', abs(np.mean(NN_score)),']')
print('[Neural Network] (Cross-Validation) Score Variance  [', np.var(NN_score),']')
print('[Neural Network] (Cross-Validation) Expected score [', abs(np.mean(NN_score)) - np.var(NN_score) ,',', abs(np.mean(NN_score)) + np.var(NN_score),']')


#reduce dimensions using Principle Component Analysis
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,2,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
sklearn_pca=PCA(n_components=1)
X=sklearn_pca.fit_transform(X)



#partition dataset 
(x_train, x_test,y_train,y_test) = train_test_split(X,Y, test_size=0.1, random_state=40)
input_dim = 1
non_linear_model = learning_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=300)
non_linear_model.fit(x =x_train,y=y_train,epochs=epochs, batch_size=batches,validation_data=(x_test,y_test),callbacks=[es])
y_pred_test = non_linear_model.predict(x_test)
y_pred_train = non_linear_model.predict(x_train)
y_pred = non_linear_model.predict(X)
np.reshape(y_pred,(1,-1))
np.reshape(Y,(1,-1))

print('[Neural Network] : MSE on test set = ',((y_test - y_pred_test) ** 2).mean())
print('[Neural Network] : MSE on training set =' ,((y_train - y_pred_train) ** 2).mean())
print('[Neural Network] : MSE error on whole set =' ,((Y - y_pred) ** 2).mean())
#score=r2_score(y_true=Y,y_pred=y_pred)
#print('[Network Network] : Accuracy on whole set = ',score)

#save trained keras model in a .pickle format
models.save_model(non_linear_model, 'non_linear_model.pickle')


print('==================================================================================')
print('==================================================================================\n\n')

# Compute t-test

print('==================================================================================')
print('========================   MODEL COMPARISON    ===================================')
print('[Neural Network](Cross Validaton) score = ',abs(np.mean(NN_score)))
print('[Linear Model]  (Cross Validaton) = ',abs(np.mean(scores_lr)))

from scipy.stats import ttest_ind
_, p_val = ttest_ind(NN_score,scores_lr)
print('\nProbability that [Neural Network] has the same mean accuracy as the [Linear Model]: {:.4f}'.format(p_val))
if p_val < 0.001:
  print('\tWith strong statistical significance, the non-linear model does not have the same accuracy as the linear model')
  if np.mean(NN_score) > np.mean(scores_lr):
    print('\t The Neural Network regularly outperforms the Linear Model')
  else:
    print('\t The  Linear Model regularly outperforms the Neural Network')
elif p_val < 0.05:
  print('\tWith statistical significance, the non-linear model does not have the same accuracy as the linear model')
  if np.mean(NN_score) > np.mean(scores_lr):
    print('\t The Neural Network regularly outperforms the Linear Model')
  else:
    print('\t The  Linear Model regularly outperforms the Neural Network')
else:
  print('The two models do not have a significant difference')
print('==================================================================================')
print('==================================================================================\n')