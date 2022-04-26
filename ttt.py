import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



a = np.array([[1,2,6,2,7,5,6,8,2,9],\
             [3,5,6,2,7,5,6,8,2,4]])
b = np.array([[9,2,6,2,7,5,6,8,2,9],\
             [3,5,6,2,7,5,6,8,2,4]])
print(np.unique(a))
print(a == 1)
print(b[a == 1])