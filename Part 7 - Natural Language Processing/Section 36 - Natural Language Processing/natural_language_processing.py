# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, model_selection, naive_bayes, metrics

# Importing the dataset
df = read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    

# Cleaning the reviews and storing them
corpus = []
nltk.download('stopwords')
for i in range(1000): # there are 1000 reviews
    review_without_unwanted_characters = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review_lowercase = review_without_unwanted_characters.lower()
    review_splitted_words = review_lowercase.split()
    # set(stopwords...) is used because casting the words to set() makes the task faster\
    # for the algorithm. without set, ie, stopwords.words('english') would also have worked\
    # but using set() is recommended.
    # review_without_unnecessary_words = [word for word in review_splitted_words\
    #                                    if word not in set(stopwords.words('english'))]
    
    # instead of applying stemming in a separate line, the previous line itself can be\
    # used to stem the words when they are being added the review word list of relevant\
    # words
    ps = PorterStemmer()
    review_stemmed_and_without_unnecessary_words = [ps.stem(word) for word in review_splitted_words\
                                        if word not in set(stopwords.words('english'))]
    review_cleaned = ' '.join(review_stemmed_and_without_unnecessary_words)
    corpus.append(review_cleaned)

# =============================================================================
# # when we execute the following line, we get a matrix of size 1000x1565 which may be\
# # having irrelevant words not found in the stopwords. Irrelevant words are those which\
# # are not common in all reviews. Such words increase the sparsity of the matrix\
# # ie, such words increase the number of '0' entries in the matrix which is undesirable\
# # for any ML algorithm. To overcome this, the CountVectorizer class comes with a\
# # parameter that can limit the maximum number of features to be selected to make the\
# # feature matrix. The parameter is called max_features
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus).toarray()
# =============================================================================

vectorizer = CountVectorizer(max_features = 1500) # we need to choose what value of\
                                                # max_features would be best for the\
                                                # dataset we have
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Classification using Naive Bayes (for NLP, Naive Bayes Decision trees and Random\
# Forest algorithms are used generally

# Naive Bayes (copy pasting the implementation already done)
# Dataset splitting
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                                                  test_size = 0.20,\
                                                                  random_state = 0)
# =============================================================================
# # Feature scaling --- not required for this problem bcoz features are 0s and 1s
# scaler_X = preprocessing.StandardScaler()
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)
# =============================================================================

# Naive Bayes Classifier Model Fitting
classifier = naive_bayes.GaussianNB()
classifier = classifier.fit(X_train, y_train)

y_predictions = classifier.predict(X_test)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
print(conf_matrix)

#values of correct predictions extracted from confusion matrix
print("Accuracy on test set: (correct pred/total test egs): " + str((55+91)/200))