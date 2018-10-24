# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:42:11 2018

@author: hamed
"""

import numpy as np
import matplotlib.pyplot as plt
import colored_traceback; colored_traceback.add_hook()

#%%
file_name = "stats.npy"
stats = np.load(file_name)  #stats = ("mean_", "scale_", "min_", "max_", "var_") * 1440

mean_ = stats[0,:]
scale_ = stats[1,:]



print("scale_ = ", scale_)

plt.figure(1)

plt.subplot(211)
plt.title("mean")
plt.plot(mean_)

plt.subplot(212)
plt.title("scale")
plt.plot(scale_)

plt.show()



