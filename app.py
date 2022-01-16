import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("./honeyproduction.csv")

# Mean total production of Honey per Year.
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# X is the year column of the prod_per_year DataFrame. It is the X-Axis.
X = prod_per_year.year
X = X.values.reshape(-1, 1) # 2-Dimentional Matrix

# Y is the total production of Honey per Year. It is the Y-Axis.
y = prod_per_year.totalprod

# Plot as a Scatterplot.
plt.scatter(X, y)

# Create a linear regression model with Sci-Kit.
regr = linear_model.LinearRegression()

regr.fit(X, y) # Fit model to Data.

"""
regr.coef_[0] gives you the m, i.e Coefficient of the X which is the Gradient.
regr.intercept_ gives you the Y-Intercept, i.e. c, in the formula:
y = mx + c
"""

y_predict = regr.predict(X) # Finds out the Y-Values

# GIves you a Line Of Best Fit for the Data.
plt.plot(X, y_predict)

# Future values of X from 2013 up until 2050.
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future) # The Y-Values for the X-Values that the regr model would predict.

plt.plot(X_future, future_predict) # Plot the graph with the future values of X and y as well.
plt.show()