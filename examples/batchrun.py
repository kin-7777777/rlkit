import numpy as np
import gtimer as gt
from examples.sac_g import sac_g_func

gt.set_def_unique(False)

# model_disc_range = np.arange(start=0.05, stop=1.05, step=0.10)
# for model_disc in model_disc_range:
#     sac_g_func(0.99, model_disc)
    
sac_g_func(value_disc=0.99, model_disc=0.80, mve_horizon=1, seed=0)

# Hypothesis: For sparse reward, the model discount for gamma mve doesn't matter.
# Remember set max epochs to 800.
# Remember check rewards are not shaped.
# Set batch size 128.
# for seed in range(100, 110):
#     sac_g_func(value_disc=0.99, model_disc=0.50, mve_horizon=1, seed=seed)
#     sac_g_func(value_disc=0.99, model_disc=0.80, mve_horizon=1, seed=seed)