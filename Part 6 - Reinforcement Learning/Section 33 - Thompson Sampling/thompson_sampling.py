# Thompson Sampling

# Importing the libraries
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import random

df = read_csv('Ads_CTR_Optimisation.csv')
human_simulations = df
# Implementing Thompson Sampling
N = 10000
d = 10
ads_selected_each_round = []
number_of_reward_0 = [0] * d
number_of_reward_1 = [0] * d
total_rewards = 0

for n in range(N):
    max_random_draw = 0
    for i in range(d):
        random_draw = random.betavariate(number_of_reward_1[i] + 1, \
                                       number_of_reward_0[i] + 1)
        if random_draw > max_random_draw:
            max_random_draw = random_draw
            temp_selected_ad = i
            
    ads_selected_each_round.append(temp_selected_ad)
    reward = human_simulations.values[n, temp_selected_ad]
    if reward == 1:
        number_of_reward_1[temp_selected_ad] += 1
    else:
        number_of_reward_0[temp_selected_ad] += 1
    total_rewards += reward
    
plt.hist(ads_selected_each_round)
plt.title('Histogram for ad selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()