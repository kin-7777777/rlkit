from statistics import mean
import torch
import csv
import os
import matplotlib.pyplot as plt

# data = torch.load('data/name-of-experiment/name-of-experiment_2022_09_09_19_52_49_0000--s-0/params.pkl')
# print(data)

with open('data/gamma-0.01speed-simplified/gamma-0.01speed-simplified_2022_09_23_08_03_52_0000--s-0/progress.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     stat = []
     for row in reader:
        #  print(row)
        #  print('\n')
        stat.append(float(row['eval/Average Returns']))
        
plt.plot(stat[:])
plt.xlabel('Epochs')
plt.ylabel('eval/Average Returns')
dir = '/mnt/e/users/kin/desktop/plots'
filename = 'gamma-0.01speed-simplified_2022_09_23_08_03_52.png'
if not os.path.exists(dir):
        os.makedirs(dir)
plt.savefig(dir+'/'+filename)
plt.show()