# Artificial Neural Network

# Installing Theano & Tensorflow using Keras library (do it in Anaconda terminal)
# pip install theano
# pip install tensorflow
# pip install keras

# PART 1: DATA PREPROCESSING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Additional step: Encoding categorical data (before splitting the dataset)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Remove 1 dummy variable to avoid falling into the dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PART 2 - MAKING THE ANN

# Importing the keras libraries and packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initializing the ANN -- initial the Sequential class from the Sequential module above
classifier = Sequential()

# Dense function takes care of initializing the weights to small numbers close to 0
# input nodes = 11 independent variables
# actiivation function for hidden layers: rectifier function
# activation function for output: sigmoid function gives us probability for each customer leaving the bank

# Adding the input layer &  the first hidden layer
# Tip: choose the number of nodes in the hidden layer as the average of the number of nodes in the input layer + output layer
# If not using the tip, then you use parameter tuning by experiementing with different perameters using a cross validation set
# input_dim is a compulsory arugument to add
# Adding the first hidden layer without dropout
classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu', input_dim = 11))

# first hideen layer WITH dropout
# arguments: p = fraction of the neurons we want to drop at each iteration; rule of thumb is that you should try p = 0.1 if there's overfitting and if that doesn't reduce overfitting, try a higher value for p.
# don't try to go over p = 0.5, or there will be underfitting / no neurons to learn on.

classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu', input_dim = 11))
# Put the line of code above under any of the other neural layers if you want to add dropout to it.

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu'))

# Adding the final layer - the output layer & change the activation function from relu to sigmoid to get probabilities in the output
classifier.add(Dense(units = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
#Output that has more than 2 categories (not binary); input 3 if you have 3 categories; activation function would be softmax for output/dependent variables that have more than 2 categories

# Compiling the ANN -- add the compile method instead of the add method
# if your dependent variable has binary outcome, the log loss is binary_crossentropy
# if your dependent variable has more than 2 outcomes, the log loss is categorical_crossentropy
# accuracy metric used with metrics argument in a list to improve model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training Set (every time the training set goes through the ANN, it's an epoch)
# 100/100 epochs will give us the accuracy score of the TRAINING set.
# no rule of thumb for batch_size or nb_epochs..art of deep learning, you get to choose!
# flacutations in accuracy scores every time you run due to stochasticity
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

# PART 3 - MAKING THE PREDICTIONS AND EVALUATING THE MODEL

# Predicting the Test set results (2000 customers at bank or 20% of total customer dataset)
# The bank can use this information and rank the customers from higher probability to leave to lowest probability to leave
# Take the top 10% of the dataset where customers are most likely to leave and apply data mining techniques to find out why these customers are inclined to leave
# Take preventative measures to keep the customers with the bank
y_pred = classifier.predict(X_test)
# Tells us TRUE or FALSE for each customer leaving the bank in the y_pred dataset.
y_pred = (y_pred > 0.5)

"""Predict if a customer not in the dataset with the following information will leave the bank
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""
# For the predict method, create a 2D array with 2 square brackets for np.array & make sure the variables above are in the same order as the array
# not standardized to the same scale yet, so apply the same scale by taking the same sc object and applying it to the code below
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0 , 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Check new_prediction table. False! the customer doesn't leave the bank.

# Making the Confusion Matrix
# Executing the equation (1541+139)/2000 in the console will give you the accuracy score of the TEST set.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Notes: Evaluating the model's performance on one single test set is not the most relevant way.
# By retraining the model multiple times, we get different accuracy scores not only on the training set but also on the test set. --> Bias-Variance Trade-off.
# The solution to the problem of changing accuracy scores is to look at model evalution, improvement and tuning techniques --> K-fold cross validation.
# K-fold cross validation will fix the high variance problem of different accuracy scores between test sets that are run over and over again.
# normally, the training set is split into 10 folds and we train it with 9 folds while testing it with the last/10th fold.
# now we can take an average of the different accuracy scores of the 10 test sets and integrate the stdev to find the variance.

# PART 4 - EVALUATING, IMPROVING & TUNING THE ANN

# Evaluating the ANN
# Use keras wrapper to convert sklearn to keras library to use the classifier we already have
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# cv = 10 is the unsaid standard and n_jobs = number of CPUs to use
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed in deep learning
# Dropout is applied to the neurons so that some of them randomly become disabled at each iteration.
# You should apply Dropout to all the neural layers when trying to avoid overfitting

# Overfitting occurs when the model was trained too much on the training set --> the model learned too much
# overfitting generally happens when you have a much higher accuracy on the training set than the test set.
# you will notice overfitting if you see a high variance when applying k-fold cross validation

# Tuning the ANN (parameter tuning)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
# tune different hyperparameters; common practice to take powers of 2
# create a new hyperparameter of the build classifier function for arguments with values already (like the optimizer), unlike the nb_epochs argument that didn't initially have a value
# to tune any parameters within the architecture of your ANN (i.e. optimizer) -- in this case, replace 'adam' with 'optimizer' (the name of the argument)
# rmsprop optimizer is normally used in deep learning for RNNs
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
