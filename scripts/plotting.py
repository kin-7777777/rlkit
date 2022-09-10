from statistics import mean
import torch
import csv
import matplotlib.pyplot as plt

# data = torch.load('data/name-of-experiment/name-of-experiment_2022_09_09_19_52_49_0000--s-0/params.pkl')

# print(data)

with open('data/name-of-experiment/name-of-experiment_2022_09_09_19_52_49_0000--s-0/progress.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     mean_rewards = []
     for row in reader:
        #  print(row)
        #  print('\n')
        mean_rewards.append(float(row['eval/Returns Mean']))
        
plt.plot(mean_rewards)
plt.show()