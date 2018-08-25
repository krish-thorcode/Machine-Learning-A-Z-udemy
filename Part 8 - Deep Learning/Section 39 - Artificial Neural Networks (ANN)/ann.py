# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential # used to initialise the ANN
from keras.layers import Dense # required to build the layers of our ANN

df = read_csv('Churn_Modelling.csv')

X = df.iloc[:, 3:13].values
y = df.iloc[:, -1].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,\
                                                    random_state = 0)

# Feature scaling- feature scaling is an absolute necessity in Deep Neural Networks\
# because ANN involves very intense computations
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Part II: Making an ANN

# First step of creating our first ANN- first step consists initialising the ANN, we'll be defining\
# it as a sequence of layers. There are two ways of initialising a deep learning model- either by\
# defining it as a sequnce of layers or the other way, defining a graph. Here we'll adopt the first\
# definition of ANN.

# Initialise the ANN (as  sequence of layers!). Since we are going to use the ANN as a classifer\
# so..
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',\
                     input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer -- if there are more than two categories, we need to have\
# number of o/p's = number of classes and also, activation = 'softmax' which is the\
# sigmoid function applied for a multi-class classification problem (dependent var can\
# have 3 possible outcomes)
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# We now have added the layers of the ANN. Now we need to compile the ANN which involves\
# configuring the process of fitting the ANN. We are going to do this with the\
# following single line of code.
# params: optimizer = 'adam' which is a gradient descent algorithm\
# loss = 'binary_crossentropy' is the loss function for sigmoid function and ANN that does\
# binary classfication. For ANN that does multiclass classification, we use loss=\
# 'categorical_entropy', metrics = ['accuracy'] which is the criterion to evaluate the\
# model. Typically we use the accuracy criterion 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
# params: X_train and y_train are the usual training dataset. batch_size is the number\
# of examples after which we would want our ANN to update the weights. nb_epoch is the\
# number of times we want the ANN to run over the same training example to optimize its\
# weight parameters. These two selections are feats of art which are decided by experime\
# nting again and again. For now, I will be following the values used in the tutorial.
# batch_size = 1 => stochastic gradient descent, we shall use something b/w batch\
# descent and stochastic gradient descent.
# following line is not what you should do because unlike other classifiers we have done till now\
# (which returned the fitted model), the ANN returns the history of the training, ie,
# loss, accuracy, etc. If we write classifier = classifer(...), the classifier var \
# will then contain a History object returned by the fit() method. So, we either should\
# keep a separate variable to store the history object or we should not keep anything on\
# the LHS, atleast not the classifier variable
# classifier = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Testing the ANN
y_predicted_probabilities = classifier.predict(X_test)
y_predictions = y_predicted_probabilities > 0.5

conf_matrix = confusion_matrix(y_test, y_predictions)