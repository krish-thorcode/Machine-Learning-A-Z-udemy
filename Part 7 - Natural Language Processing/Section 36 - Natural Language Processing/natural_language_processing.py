# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
