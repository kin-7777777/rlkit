from statistics import mean
import torch
import csv
import matplotlib.pyplot as plt

# data = torch.load('data/name-of-experiment/name-of-experiment_2022_09_09_19_52_49_0000--s-0/params.pkl')
# print(data)

with open('data/fixed-speed-sac/fixed-speed-sac_2022_09_10_19_36_02_0000--s-0/progress.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     stat = []
     for row in reader:
        #  print(row)
        #  print('\n')
        stat.append(float(row['expl/path length Max']))
        
plt.plot(stat[115:])
plt.show()