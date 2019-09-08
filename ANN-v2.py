# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask
from flask import render_template
from flask import request
from flask_ngrok import run_with_ngrok

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
#new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#new_prediction = (new_prediction > 0.5)
#print(new_prediction)

def predict_helper(a1, a2, b, c, d, e, f, g, h, i, j):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)

    # Part 3 - Making predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    output = classifier.predict(sc.transform(np.array([[a1, a2, b, c, d, e, f, g, h, i, j]])))
    return output

#print(predict_helper(0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000))

# Hosting on a server
# creates a Flask application, named app
app = Flask(__name__)
run_with_ngrok(app)

# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict")
def predict():
    geography = request.args.get('geography')
    credit_score = request.args.get('creditScore')
    gender = request.args.get('gender')
    age = request.args.get('age')
    tenure = request.args.get('tenure')
    balance = request.args.get('balance')
    products = request.args.get('products')
    has_credit_card = request.args.get('hasCreditCard')
    is_active_member = request.args.get('isActiveMember')
    salary = request.args.get('salary')
    print(geography, credit_score, salary)
    #new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, credit_score, gender, age, tenure, balance, products, has_credit_card, is_active_member, salary]])))
    #new_prediction = (new_prediction > 0.5)
    #new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
    new_prediction = predict_helper(0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000)
    print(new_prediction[0][0])
    return (str(new_prediction[0][0]))

#new_prediction = predict_helper(0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000)
#print(new_prediction)

# run the application
if __name__ == "__main__":
    app.run()
