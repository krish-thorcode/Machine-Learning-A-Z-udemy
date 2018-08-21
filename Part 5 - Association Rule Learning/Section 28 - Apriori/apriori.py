# Apriori

# importing the modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from apyori import apriori

df = read_csv('Market_Basket_Optimisation.csv', header = None)

transactions_nparray = df.values
transactions_pylist = transactions_nparray.tolist() # transactions is a python list
transactions = []
for transaction in transactions_pylist:
    transactions.append([str(item) for item in transaction])
# =============================================================================
# transactions = []
# for i in range(0,7501):
#     transactions.append([str(df.values[i, j]) for j in range(0, 20)])
#     
# =============================================================================
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3,\
                min_length = 2)

# Visualising the results
results = list(rules) # the apriori function we've used from apyori module does the\
                    # sorting automatically. This sorting is not based on the normally\
                    # used lift value but on a combination of the confidence, support and\
                    # lift values
