import numpy as np
import gtimer as gt
from examples.sac_g import sac_g_func

gt.set_def_unique(False)

model_disc_range = np.arange(start=0.05, stop=1.05, step=0.10)
for model_disc in model_disc_range:
    sac_g_func(0.99, model_disc)