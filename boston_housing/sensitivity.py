import numpy as np
import pandas as pd

trial_values = np.array (
    [391183.33, 391183.33, 419700.00,
     415800.00, 420622.22, 413334.78,
     411931.58, 399663.16, 407232.00,
     351577.61, 413700.00])

trial_values_min = trial_values.min ()
trial_values_max = trial_values.max ()
trial_values_min_max_range = trial_values_max - trial_values_min


print ('Min: {:.2f}'.format (trial_values_min))
print ('Max: {:.2f}'.format (trial_values_max))
print ('Range: {:.2f}'.format (trial_values_min_max_range))
print ('Mean: {:.2f}'.format (trial_values.mean ()))

print ('Range in % of Min: {:.2%}'.format (trial_values_min_max_range/trial_values_min))
print ('Range in % of Max: {:.2%}'.format (trial_values_min_max_range/trial_values_max))