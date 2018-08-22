# Upper Confidence Bound

# Importing the libraries
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import math

df= read_csv('Ads_CTR_Optimisation.csv')
human_simulation = df # each row in the dataset is a simulation of a users. since\
                    # we cannot throw ads on real human users at this point, since we\
                    # do not have a website, and even if we did, we might not have the \
                    # traffic to have sufficient people! so, each row is a user, and\
                    # the 0's and 1's in a particular row represent whether that user\
                    # would click a particular ad. For eg, 0 under column 'Ad2' and row 2\
                    # means that the user represented by row 2 does not like Ad2, and\
                    # does not click. This is not the dataset that we are using to\
                    # compute anything for the ucb algorithms; this dataset represents\
                    # users and the choices they would make.
                    # in the context of multiarm problem, this dataset represents\
                    # persons who went to casino and will play the casino game

# Implementing UCB
N = 10000 # number of users, ie, number of rounds
d = 10 # number of ads,ie, number of arms
number_of_selections = [0] * d
sum_of_rewards = [0] * d
ad_selected_each_round = [] # this var is not needed for the algorithm, it is for\
                            # checking how the ads get selected after each round \
                            # ie, it is basically to check that our algo is actually\
                            # doing the task well.
total_reward = 0
for n in range(N):
    max_upper_bound= 0
    temp_selected_ad = 0
    for i in range(d):
        # the following if-else block makes sure that each of the 10 ads get selected\
        # by the algorithm once, ie, each ad is thrown at a user atleast once. We hence,\
        # allow the first ten users to receive each ad one-by-one, and to make that\
        # possible, if-else block has been used. It is upto the user whether it clicks\
        # on it or not.This will help the algo get the recent behaviour so that it can\
        # decide which ad should be thrown next
         
        if number_of_selections[i] > 0:
            average_reward = sum_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt((3*math.log(n + 1))/(2*number_of_selections[i]))
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            temp_selected_ad = i
        
    ad_selected_each_round.append(temp_selected_ad)
    number_of_selections[temp_selected_ad] += 1
    reward = human_simulation.values[n, temp_selected_ad] # reward = 0, if user didn't clk
    sum_of_rewards[temp_selected_ad] += reward
    total_reward += reward

# Visualising the result with histograms
plt.hist(ad_selected_each_round)
plt.title('Histogram of number of times each ad got selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()