# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# change the directory to csv
import os
os.chdir("E:\Machine learning\csv")

#import the data set
dataset = pd.read_csv('Blogging_Income.csv')

# With iloc() function, we can retrieve a particular value belonging to a row
# and column using the index values assigned to it.
# dataframe.iloc[:,start_col:end_col]
# data.iloc[:, 0:2] # first two columns of data frame with all rows
# Ref: https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
X = dataset.iloc[:, :-1].values #independent variable # select all col w/o last col
y = dataset.iloc[:, 1].values #dependent variable #select 2nd col

## split your dataset into a training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) #when you re-run your code, random_state = 0, this ensures that we all get the same results.


# fit simple linear regression to the training set to learn correlations.
# Basically, we’re going to get our computer to learn the correlations in our training set
# so that it can predict the dependent variable (income) based on the independent variable (blogging experience).
# It will create the ML model
# LinearRegression algorithm is applied to train set to prepare a model and fit that model to learn the computer
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict test set results
y_pred = regressor.predict(X_test)


# visualize training set results
plt.scatter(X_train, y_train, color = '#fe5656')
plt.plot(X_train, regressor.predict(X_train), color = '#302a2c')
plt.title('Income vs Experience (Training)')
plt.xlabel('Months of Experience')
plt.ylabel('Income')
plt.show()


# visualize test set results
plt.scatter(X_test, y_test, color = '#fe5656')
plt.plot(X_train, regressor.predict(X_train), color = '#302a2c')
plt.title('Income vs Experience (Test)')
plt.xlabel('Months of Experience')
plt.ylabel('Income')
plt.show()