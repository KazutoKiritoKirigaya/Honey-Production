import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("./honeyproduction.csv")

# Mean total production of Honey per Year.
prod_per_year = df.groupby('year').totalprod.mean().reset_index()