import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def alpha_n(curly_g, g, n):
    return ((1 - curly_g)*(curly_g - g)**(n-1)) / (1 - g)**n

# To check the claim that mve horizon of 1 with discount 0.8 is equivalent to model rollout of 5
zero_rollout = 0
for i in range(1,6):
    zero_rollout += alpha_n(0.99, 0, i)
mve_rollout = alpha_n(0.99, 0.8, 1)
print(zero_rollout)
print(mve_rollout)

# To plot Figure 2b in the paper
for value_discount in [0.5, 0.75, 0.9, 0.95, 0.975, 0.99]:
    
    model_discount_range = np.linspace(start=0, stop=value_discount, num=int(value_discount/0.001)+1)

    y_axis = []
    for model_discount in model_discount_range:
        for i in range(1, 1000):
            if np.sum(alpha_n(value_discount, model_discount, np.array(np.arange(start=1, stop=i+1)))) >= 0.95:
                y_axis.append(i)
                break
    y_axis = np.array(y_axis)

    plt.plot(model_discount_range, y_axis)
    
plt.yscale("log")
# plt.ylim(10**0, 10**3)

plt.show()