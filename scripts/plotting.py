from statistics import mean
import torch
import csv
import matplotlib.pyplot as plt

# data = torch.load('data/name-of-experiment/name-of-experiment_2022_09_09_19_52_49_0000--s-0/params.pkl')
# print(data)

with open('data/gamma-test-0.01speed/gamma_test_0.01speed_2022_09_16_10_29_01_0000--s-0/progress.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     stat = []
     for row in reader:
        #  print(row)
        #  print('\n')
        stat.append(float(row['eval/Average Returns']))
        
plt.plot(stat[:300])
plt.xlabel('Epochs')
plt.ylabel('Eval/Average Returns')
plt.savefig('/mnt/e/users/kin/desktop/plots/gamma_test_0.01speed_2022_09_16_10_29_01.png')
plt.show()